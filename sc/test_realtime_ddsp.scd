/*
Real-time OSC test controller for SensorDDSP (realtime_ddsp.py).

Sends proximity sensor values (0.0..1.0) at 100 Hz (matching Sentire's
controlRate and interactive_ddsp's 48000 Hz / 480-sample block rate) to
realtime_ddsp.py over OSC on 127.0.0.1:9000.

Sentire conventions (from sc/record_dataset_v5.scd):
  - controlRate:    100 Hz (0.01 s)
  - thresholdTouch: 0.98 (values >= 0.98 trigger touch state)
  - 4-sound V9:     ["01_canopy", "02_forestPad", "03_harmonicSwarm", "05_moss_and_mist"]
  - 5-sound V6-V8c: adds "04_modularPercussions" at index 3

How to use:
  1. Start the Python real-time DDSP server from the interactive_ddsp repo root:
       .venv/bin/python realtime_ddsp.py --run <run_dir> --ckpt state_final.pth
  2. In SuperCollider, set `useFiveSounds` below (false for V9, true for V6-V8c).
  3. Evaluate the main block below (Cmd+Enter / Ctrl+Enter inside the outer parens).
  4. Close the window (or re-evaluate the block) for clean teardown.
*/

(
var addr, controlRate, dt, thresholdTouch;
var useFiveSounds, sounds;
var sensorVal, touchHeld, soundId, noteHz, gainVal;
var win, updateSensorText, updateNoteText, updateGainText;
var sensorSlider, sensorText, touchBtn, soundMenu;
var noteSlider, noteText, baseF0Btn;
var gainSlider, gainText, resetBtn;

// --- Safe re-evaluation: stop previous routine and close previous window first ---
if(~ddspRoutine.notNil) {
	~ddspRoutine.stop;
	~ddspRoutine = nil;
};
if(~ddspWin.notNil and: { ~ddspWin.isClosed.not }) {
	~ddspWin.close;
};

// --- Configuration ---
addr = NetAddr("127.0.0.1", 9000);
controlRate = 100; // Hz (48000 / 480)
dt = controlRate.reciprocal;
thresholdTouch = 0.98;

// Toggle checkpoint sound table:
//   false -> 4-sound V9 checkpoints (indices 0..3)
//   true  -> 5-sound V6-V8c checkpoints (adds 04_modularPercussions at index 3)
useFiveSounds = false;

sounds = if(useFiveSounds) {
	["01_canopy", "02_forestPad", "03_harmonicSwarm", "04_modularPercussions", "05_moss_and_mist"]
} {
	["01_canopy", "02_forestPad", "03_harmonicSwarm", "05_moss_and_mist"]
};

// --- Initial state ---
sensorVal = 0.0;
touchHeld = false;
soundId = 0;
noteHz = 0.0; // 0.0 = use model's base f0
gainVal = 1.0;

// --- GUI Window ---
win = Window("SensorDDSP Real-Time OSC Controller", Rect(120, 120, 440, 265), resizable: false);
win.view.decorator = FlowLayout(win.view.bounds, 12@12, 8@8);
~ddspWin = win;

// Readout helpers (GUI-thread only)
updateSensorText = {
	var v = if(touchHeld) { 1.0 } { sensorVal };
	var tag = if(v >= thresholdTouch) { " [TOUCH]" } { "" };
	sensorText.string = v.round(0.001).asString ++ tag;
};

updateNoteText = {
	noteText.string = if(noteHz <= 0.0) {
		"Base f0 (0.0 Hz)"
	} {
		"% Hz".format(noteHz.round(0.1))
	};
};

updateGainText = {
	gainText.string = gainVal.round(0.01).asString;
};

// 1. Proximity slider + live readout + momentary TOUCH button
StaticText(win, Rect(0, 0, 85, 22)).string_("Proximity:");
sensorSlider = Slider(win, Rect(0, 0, 210, 22))
	.value_(sensorVal)
	.action_({ |sl|
		sensorVal = sl.value.clip(0.0, 1.0);
		updateSensorText.value;
	});
sensorText = StaticText(win, Rect(0, 0, 105, 22)).string_("0.0");

StaticText(win, Rect(0, 0, 85, 28)).string_("");
touchBtn = Button(win, Rect(0, 0, 210, 28))
	.states_([["HOLD TOUCH (1.0)", Color.white, Color.new255(180, 50, 50)]])
	.mouseDownAction_({
		touchHeld = true;
		updateSensorText.value;
	})
	.mouseUpAction_({
		touchHeld = false;
		updateSensorText.value;
	});
StaticText(win, Rect(0, 0, 105, 28)).string_(">= 0.98 = touch");

// 2. Sound selector
StaticText(win, Rect(0, 0, 85, 24)).string_("Sound:");
soundMenu = PopUpMenu(win, Rect(0, 0, 323, 24))
	.items_(sounds.collect { |name, i| "% : %".format(i, name) })
	.value_(soundId)
	.action_({ |menu|
		soundId = menu.value.asInteger;
		addr.sendMsg("/ddsp/sound", soundId);
	});

// 3. Note Hz (65-520 Hz exponential) + Use Base f0 button
StaticText(win, Rect(0, 0, 85, 22)).string_("Note Hz:");
noteSlider = Slider(win, Rect(0, 0, 210, 22))
	.value_(0.0)
	.action_({ |sl|
		noteHz = sl.value.linexp(0.0, 1.0, 65.0, 520.0);
		updateNoteText.value;
		addr.sendMsg("/ddsp/note", noteHz.asFloat);
	});
noteText = StaticText(win, Rect(0, 0, 105, 22)).string_("Base f0 (0.0 Hz)");

StaticText(win, Rect(0, 0, 85, 24)).string_("");
baseF0Btn = Button(win, Rect(0, 0, 210, 24))
	.states_([["Use Base f0 (0.0)"]])
	.action_({
		noteHz = 0.0;
		updateNoteText.value;
		addr.sendMsg("/ddsp/note", 0.0);
	});
StaticText(win, Rect(0, 0, 105, 24)).string_("");

// 4. Gain slider (0.0..2.0) + Reset State button
StaticText(win, Rect(0, 0, 85, 22)).string_("Gain:");
gainSlider = Slider(win, Rect(0, 0, 210, 22))
	.value_(gainVal.linlin(0.0, 2.0, 0.0, 1.0))
	.action_({ |sl|
		gainVal = sl.value.linlin(0.0, 1.0, 0.0, 2.0);
		updateGainText.value;
		addr.sendMsg("/ddsp/gain", gainVal.asFloat);
	});
gainText = StaticText(win, Rect(0, 0, 105, 22)).string_("1.0");

StaticText(win, Rect(0, 0, 85, 28)).string_("");
resetBtn = Button(win, Rect(0, 0, 210, 28))
	.states_([["Reset State (/ddsp/reset)"]])
	.action_({
		addr.sendMsg("/ddsp/reset");
		"DDSP: sent /ddsp/reset".postln;
	});

// --- Clean teardown on window close ---
win.onClose_({
	if(~ddspRoutine.notNil) {
		~ddspRoutine.stop;
		~ddspRoutine = nil;
	};
	"DDSP OSC controller stopped.".postln;
});

// Send initial parameters once, then stream /ddsp/sensor at 100 Hz on SystemClock
addr.sendMsg("/ddsp/sound", soundId);
addr.sendMsg("/ddsp/note", noteHz.asFloat);
addr.sendMsg("/ddsp/gain", gainVal.asFloat);

~ddspRoutine = Routine({
	loop {
		var val = if(touchHeld) { 1.0 } { sensorVal };
		addr.sendMsg("/ddsp/sensor", val.asFloat, soundId.asInteger, noteHz.asFloat);
		dt.wait;
	};
}).play(SystemClock);

updateSensorText.value;
updateNoteText.value;
updateGainText.value;
win.front;

"DDSP OSC controller running -> 127.0.0.1:9000 at % Hz (% sounds)".format(controlRate, sounds.size).postln;
)


/*
===============================================================================
OPTIONAL: Sentire live bus tap (100 Hz -> /ddsp/sensor)
===============================================================================
Use this instead of the GUI slider when Sentire is booted (g.controller exists)
to forward Sentire's live post-scaling proximity signal (0..1) directly to
realtime_ddsp.py at 100 Hz (48000 Hz sample rate).

Evaluate the block below after Sentire has finished booting:

(
var ddspAddr = NetAddr("127.0.0.1", 9000);
var proxNode, proxBus;

if(g.isNil or: { g.controller.isNil }) {
	"Sentire not booted: g.controller not found.".error;
} {
	// Stop GUI routine if running so only one source drives /ddsp/sensor
	if(~ddspRoutine.notNil) { ~ddspRoutine.stop; ~ddspRoutine = nil };

	// Tap Sentire's post-scaling proximity bus (audio-rate in Sentire signalFlow)
	proxNode = g.controller.signalFlow.proximityDetection;
	proxBus = proxNode.bus.index; // use proxNode.proxy.bus.index if wrapped in a proxy object

	~ddspSoundId = 0;  // change live: ~ddspSoundId = 2;
	~ddspNoteHz = 0.0; // change live: ~ddspNoteHz = 220.0;

	SynthDef(\ddspSentireTap, { |inBus = 0, rate = 100|
		var sig = A2K.kr(In.ar(inBus, 1)).clip(0.0, 1.0);
		SendReply.kr(Impulse.kr(rate), '/sentire_prox', sig);
	}).add;

	Server.default.bind {
		Server.default.sync;
		if(~ddspTapSynth.notNil) { ~ddspTapSynth.free };
		~ddspTapSynth = Synth.tail(Server.default, \ddspSentireTap, [
			\inBus, proxBus,
			\rate, 100
		]);
	};

	OSCdef(\sentireToDdsp, { |msg|
		var sensorVal = msg[3].asFloat.clip(0.0, 1.0);
		ddspAddr.sendMsg("/ddsp/sensor", sensorVal, ~ddspSoundId.asInteger, ~ddspNoteHz.asFloat);
	}, '/sentire_prox');

	"Sentire live tap active on bus % -> 127.0.0.1:9000 (/ddsp/sensor @ 100 Hz)".format(proxBus).postln;
};
)

// Teardown for Sentire live bus tap:
(
if(~ddspTapSynth.notNil) { ~ddspTapSynth.free; ~ddspTapSynth = nil };
OSCdef(\sentireToDdsp).free;
"Sentire live tap stopped.".postln;
)
*/
