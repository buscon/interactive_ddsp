/*
Records the training dataset for interactive_ddsp: the SAME set of random proximity
curves is replayed through several Sentire sound environments, and each
replay is recorded.

Why it is built this way (see SENSOR_SONIFICATION_PLAN.md, "Multi-Sound Replay"):
  - Every sound gets exactly the same proximity curves. The proximity curve file name is the
    key, so train/validation can be split by proximity curve (no gesture in train with one
    sound and in validation with another).
  - A "proximity curve" here is only the sensor value over time (0..1, one curve per
    recording), not a spatial path. The curves are random but reproducible
    (settings[\seed]) and long (60 s). They mix holds, ramps of different speeds, and
    touches of short and long duration (value = 1), so the model sees many more
    different time courses than a fixed sweep/touch series. They can be cut into 2 s
    training chunks later.
  - The sensor value is written to disk at a fixed control rate (100 Hz = 48000 / 480,
    the control rate in config.yaml), next to the audio.

How to use:
  1. Boot Sentire as usual (main.scd or main-headless.scd), so that g.controller exists.
  2. Adjust the settings below if needed.
  3. Evaluate the whole block (Cmd+Enter inside the parentheses).
  4. Stop early with: ~stopRecordingSeries.value;
  5. To resume an interrupted run, set `folder` to the existing folder: files that are
     already recorded are skipped (the trajectories are regenerated identically from the seed).

While it runs, the sensor signal is replaced by a simulated one (injected after the
scaling stage, so the sensor input is ignored). When it is finished, or stopped, the
sensor signal and the sound environment that was selected before are restored.

Output, in <sentire folder>/Recordings/Dataset_<date>/ (or `folder`):
  proximity_curves/proximity_curve_001.csv ...   time_s,sensor   (sensor = 0..1 at controlRate)
  <sound>_<proximity_curve>_<node>.aiff      audio (int16, as in NodeRecorder)
  manifest.csv    one row per audio file: sound, proximity_curve, file base, proximity_curve file,
                  proximity_curve_start_s (time of the proximity curve start inside the audio file),
                  duration_s
  events.csv      recording_start / proximity_curve_start / proximity_curve_end / recording_end
                  times in seconds since the start of each file
  dataset_info.txt  seed, rates, sounds, proximity_curve settings

Limits:
  - There is ONE sensor channel: Sentire's single post-scaling signal, 0..1. The value
    reaching thresholdTouch (0.98) is the touch event, and while it stays at or above it
    the sound changes with the seconds spent in touch. "Seconds in touch" is not in the
    csv; derive it from the sensor column (count while the value is >= 0.98) in
    preprocessing, and the same way live.
  - proximity_curve_start_s is measured in the language, so it is accurate to a few ms, not
    to the sample. It is good enough for a 100 Hz control rate, but the sync check
    (issue "Record replay dataset with sync check") should confirm it.
  - This script has not been run against a live Sentire setup yet; do a short test run
    first (nProximityCurves: 2, proximityCurveDur: 10, one sound).
*/
(
var settings = (
	// [name used in the file names, key of the sound environment]
	sounds: [
		["01_canopy", \whisperingCanopy],
		["02_forestPad", \forestPad],
		["03_harmonicSwarm", \harmonicSwarm],
		["04_modularPercussions", \modularPercussiveCloud],
		["05_moss_and_mist", \mossAndMist]
	],

	// Nodes of the signal flow to record. One file per node.
	// Options: \master (what you hear), \effects, \soundEnvironment, \proximityDetection, \rawSignal
	nodes: [\master],

	// Output folder. nil: <sentire folder>/Recordings/Dataset_<date>
	folder: nil,
	skipExisting: true, // skip a file that already exists (for resuming in a given folder)

	seed: 2026,           // same seed -> same proximity curves
	nProximityCurves: 20,    // unique proximity curves. 20 x 60 s = 600 chunks of 2 s
	proximityCurveDur: 60,    // s, long enough to contain holds of 30 s and more
	// A touch is the signal reaching thresholdTouch (Sentire's value); the sound then changes
	// with the seconds spent in touch. Random ramps stay at or below 0.95, so they never touch.
	thresholdTouch: 0.98,
	touchHoldShort: [0.3, 3.0],   // s, short touches: mostly only the touch sound, little stay-in-touch change
	touchHoldLong: [5.0, 45.0],   // s, long holds: the sound keeps changing over about 30 s
	pLongTouch: 0.5,              // probability that a touch is a long hold
	controlRate: 100,     // Hz, sensor values written to the csv. = sampling_rate / block_size

	settle: 2,      // s, simulated signal is held at 0 before the recording starts
	lead: 1,        // s, recorded before the proximity curve starts
	tail: 5,        // s, recorded after the proximity curve ends (to capture the release of the sound)
	touchRise: 0.1  // s, time the simulated signal needs to go from 0 to 1 (and back) for a touch
);

var sentire, server, originalSoundKey;
var processing, ctl;
var controlFunction, makeProximityCurve, writeProximityCurve, cleanup, peakOf, writeEvents, writeManifest, writeInfo;
var folder, eventRows = List[], manifestRows = List[], buffers = List[], recorder = nil, running = false;

// Check if Sentire is booted
if (g.isNil) {
	"Dataset: g not found, boot Sentire first (main.scd or main-headless.scd)".error;
	^this
};

sentire = g.controller;
if (sentire.isNil) {
	"Dataset: g.controller not found, boot Sentire first".error;
	^this
};

server = Server.default;

// the shared simulated signal: plays a buffer of sensor values at `rate` values per second,
// started by t_go. After the end of the buffer the last value is held.
controlFunction = {
	var trig = \t_go.tr(0);
	var buf = \buf.kr(0);
	var rate = \rate.kr(100);
	var pos = Sweep.kr(trig, rate).clip(0, BufFrames.kr(buf) - 1);
	K2A.ar(BufRd.kr(1, buf, pos, 0, 2));
};

// Random proximity curve in 0..1. Starts and ends at 0.
// Segments: hold (stay), ramp (go to a random level at a random speed), touch (go to 1, hold, back to 0).
makeProximityCurve = { arg dur, rate, touchRise, holdShort, holdLong, pLong;
	var levels = List[0.0], times = List[], t = 0, env;
	var add = { arg dt, level;
		var step = dt.min(dur - 1 - t); // leave at least 1 s for the final ramp to 0
		if (step > 0.001) {
			times.add(step);
			levels.add(level);
			t = t + step;
		};
	};
	while { t < (dur - 1.001) } {
		switch ([\hold, \ramp, \touch].wchoose([0.3, 0.5, 0.2]))
		{ \hold } { add.(exprand(0.5, 6.0), levels.last) }
		{ \ramp } { add.(exprand(0.3, 10.0), if (0.15.coin) { 0.0 } { rrand(0.0, 0.95) }) }
		{ \touch } {
			add.(touchRise, 1.0);
			// short touches and long holds, so the sound's change with time-in-touch is covered
			add.(if (pLong.coin) { rrand(holdLong[0], holdLong[1]) } { rrand(holdShort[0], holdShort[1]) }, 1.0);
			add.(touchRise, 0.0);
		};
	};
	times.add(dur - t);
	levels.add(0.0);
	env = Env(levels.asArray, times.asArray, \lin);
	FloatArray.fill((dur * rate).asInteger, { arg i; env.at(i / rate) });
};

writeProximityCurve = { arg path, values, rate;
	File.use(path.standardizePath, "w", { arg file;
		file.write("time_s,sensor\n");
		values.do { arg v, i; file.write("%,%\n".format((i / rate).round(0.0001), v.round(0.0001))) };
	});
};

peakOf = { arg path;
	var sf = SoundFile.openRead(path);
	var info = nil;
	if (sf.notNil) {
		var data = FloatArray.newClear(sf.numFrames * sf.numChannels);
		sf.readData(data);
		info = (duration: sf.duration, peak: data.abs.maxItem);
		sf.close;
	};
	info;
};

writeEvents = {
	File.use((folder +/+ "events.csv").standardizePath, "w", { arg file;
		file.write("sound,proximity_curve,file,event,time_s\n");
		eventRows.do { arg row; file.write(row.join(",") ++ "\n") };
	});
};

writeManifest = {
	File.use((folder +/+ "manifest.csv").standardizePath, "w", { arg file;
		file.write("sound,proximity_curve,file,proximity_curve_file,proximity_curve_start_s,duration_s\n");
		manifestRows.do { arg row; file.write(row.join(",") ++ "\n") };
	});
};

writeInfo = {
	File.use((folder +/+ "dataset_info.txt").standardizePath, "w", { arg file;
		file.write("seed: %\n".format(settings[\seed]));
		file.write("nTrajectories: %\n".format(settings[\nTrajectories]));
		file.write("proximityCurveDur_s: %\n".format(settings[\proximityCurveDur]));
		file.write("controlRate_Hz: %\n".format(settings[\controlRate]));
		file.write("lead_s: %\n".format(settings[\lead]));
		file.write("tail_s: %\n".format(settings[\tail]));
		file.write("touchRise_s: %\n".format(settings[\touchRise]));
		file.write("thresholdTouch: %\n".format(settings[\thresholdTouch]));
		file.write("touchHoldShort_s: %\n".format(settings[\touchHoldShort]));
		file.write("touchHoldLong_s: %\n".format(settings[\touchHoldLong]));
		file.write("pLongTouch: %\n".format(settings[\pLongTouch]));
		file.write("nodes: %\n".format(settings[\nodes]));
		file.write("sounds: %\n".format(settings[\sounds].collect { arg s; s[0] }));
	});
};

cleanup = {
	running = false;
	if (recorder.notNil) {
		try { if (recorder.isRecording) { recorder.stopRecording } };
		try { recorder.clear };
		recorder = nil;
	};
	try { sentire.resetSignalSource };
	if (originalSoundKey.notNil) {
		try { sentire.soundEnvironmentSelector.selected = originalSoundKey };
	};
	buffers.do { arg b; try { b.free } };
	buffers.clear;
	"Dataset: signal source and sound environment restored".postln;
};

~stopRecordingSeries = {
	if (~recordingSeries.notNil) { ~recordingSeries.stop };
	cleanup.value;
};

if (~recordingSeries.notNil) { ~recordingSeries.stop };
running = true;
originalSoundKey = sentire.soundEnvironmentSelector.selected;
folder = settings[\folder] ?? {
	g.config.sentireFolder +/+ "Recordings" +/+ "Dataset_" ++ Date.getDate.format("%Y%m%d_%H%M%S")
};
File.mkdir(folder);
File.mkdir(folder +/+ "proximity_curves");

	~recordingSeries = fork {
		try {
			var nodes = (), trajectories = List[], silentBuf, rate = settings[\controlRate];
			var perFile = settings[\settle] + settings[\lead] + settings[\proximityCurveDur] + settings[\tail] + 1;

			settings[\nodes].do { arg name;
				nodes[name] = sentire.signalFlow.perform(name);
			};

			// generate all trajectories once; every sound replays the same ones
			thisThread.randSeed = settings[\seed];
			settings[\nProximityCurves].do { arg i;
				var name = "proximity_curve_%".format((i + 1).asString.padLeft(3, "0"));
				var values = makeProximityCurve.(settings[\proximityCurveDur], rate, settings[\touchRise],
					settings[\touchHoldShort], settings[\touchHoldLong], settings[\pLongTouch]);
				var buffer = Buffer.sendCollection(server, values, 1);
				buffers.add(buffer);
				writeProximityCurve.(folder +/+ "proximity_curves" +/+ (name ++ ".csv"), values, rate);
				trajectories.add((name: name, buffer: buffer, dur: settings[\proximityCurveDur]));
			};
			silentBuf = Buffer.sendCollection(server, FloatArray[0, 0], 1);
			buffers.add(silentBuf);
			server.sync;
			writeInfo.value;

			sentire.simulateProximityCustomFunction(\soundSeries, controlFunction);
			server.sync;
			processing = sentire.signalFlow.proximityDetection.sensorProcessing;
			ctl = processing.customSignal.proxy;
			ctl.set(\buf, silentBuf.bufnum, \rate, rate, \t_go, 1);

			"Dataset: % proximity curves x % sounds = % files, about % min".format(
				settings[\nProximityCurves], settings[\sounds].size,
				settings[\nProximityCurves] * settings[\sounds].size,
				(settings[\nProximityCurves] * settings[\sounds].size * perFile / 60).round(1)).postln;

			settings[\sounds].do { arg sound;
				var soundName = sound[0], soundKey = sound[1];

				if (SentireSoundEnvironment.all[soundKey].isNil) {
					"Dataset: sound environment '%' not loaded, skipping %".format(soundKey, soundName).warn;
				} {
					sentire.soundEnvironmentSelector.selected = soundKey;
					ctl.set(\buf, silentBuf.bufnum, \t_go, 1);
					settings[\settle].wait;

					trajectories.do { arg traj;
						var base = "%_%".format(soundName, traj[\name]);
						var path = folder +/+ base ++ ".aiff";
						var firstFile = folder +/+ "%_%.aiff".format(base, nodes.keys.asArray.first);
						var waited = 0, prepared = false, t0, proximity_curveStart;
						var event = { arg label;
							eventRows.add([soundName, traj[\name], base, label, (SystemClock.seconds - t0).round(0.001)]);
						};

						if (settings[\skipExisting] and: { File.exists(firstFile) }) {
							"Dataset: skipping %, already recorded".format(base).postln;
						} {
							"Dataset: recording %".format(base).postln;

							// signal at 0, let the sound settle
							ctl.set(\buf, silentBuf.bufnum, \t_go, 1);
							settings[\settle].wait;

							recorder = NodeRecorder(path, nodes, true);
							recorder.prepareRecording({ prepared = true });
							while { prepared.not and: { waited < 10 } } { 0.1.wait; waited = waited + 0.1 };
							if (prepared.not) { Error("Recorder was not prepared for %".format(base)).throw };

							recorder.startRecording;
							t0 = SystemClock.seconds;
							event.("recording_start");
							settings[\lead].wait;

							// replay the proximity curve
							proximity_curveStart = (SystemClock.seconds - t0).round(0.001);
							event.("proximity_curve_start");
							ctl.set(\buf, traj[\buffer].bufnum, \rate, rate, \t_go, 1);
							traj[\dur].wait;
							event.("proximity_curve_end");

							settings[\tail].wait;
							event.("recording_end");
							recorder.stopRecording;
							1.wait;
							recorder.clear;
							recorder = nil;
							ctl.set(\buf, silentBuf.bufnum, \t_go, 1);

							manifestRows.add([soundName, traj[\name], base,
								"proximity_curves/" ++ traj[\name] ++ ".csv", proximity_curveStart, traj[\dur]]);

							// report what ended up on disk
							nodes.keys.do { arg nodeName;
								var file = folder +/+ "%_%.aiff".format(base, nodeName);
								var info = peakOf.(file);
								if (info.isNil) {
									"Dataset: FILE MISSING %".format(file).warn;
								} {
									"  % s, peak % dBFS  (%)".format(
										info[\duration].round(0.1), info[\peak].ampdb.round(0.1), PathName(file).fileName).postln;
									if (info[\peak] < 0.0001) { "  -> this file is silent".warn };
								};
							};
							writeEvents.value;
							writeManifest.value;
						};
					};
				};
			};
			"Dataset: done, files are in %".format(folder).postln;
		} { arg error;
			error.reportError;
			"Dataset: aborted because of the error above".error;
		};
		cleanup.value;
	};
)
