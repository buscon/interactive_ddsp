/*
Real-time SuperCollider test controller for realtime_ddsp.py.

Sends OSC messages at 100 Hz (= 48000 / 480) to Python:
  /ddsp/sensor [sensor_0_to_1, sound_id, note_hz]
  /ddsp/gain   [gain]
  /ddsp/reset

Usage:
  1. In a terminal, start the Python streaming server:
       python3 realtime_ddsp.py --run <run_dir> --ckpt state_final.pth
  2. In SuperCollider, evaluate the GUI block below (Cmd+Enter / Ctrl+Enter).
  3. Move the Proximity slider (push >= 0.98 or hold the TOUCH button to trigger touch).
*/

(
var addr = NetAddr("127.0.0.1", 9000);
var sensorVal = 0.0;
var touchHeld = false;
var soundId = 0;
var noteHz = 0.0; // 0.0 = use model's f0_base
var gain = 1.0;

// Update this list if using a 5-sound checkpoint (insert "04_modularPercussions" at index 3)
var soundItems = [
	"0: 01_canopy",
	"1: 02_forestPad",
	"2: 03_harmonicSwarm",
	"3: 05_moss_and_mist"
];

var win, proxSlider, proxLabel, touchBtn, soundMenu, noteSlider, noteLabel, useBaseBtn, gainSlider, resetBtn;

if (~ddspWin.notNil) { try { ~ddspWin.close } };
if (~ddspRoutine.notNil) { ~ddspRoutine.stop };

win = Window("Interactive DDSP — Real-Time Test", Rect(200, 200, 420, 260)).front;
~ddspWin = win;
win.view.decorator = FlowLayout(win.view.bounds, 12@12, 8@8);

// Proximity slider
proxLabel = StaticText(win, 396@20).string_("Proximity: 0.000");
proxSlider = Slider(win, 310@28)
	.value_(0.0)
	.action_({ |sl|
		sensorVal = sl.value;
		proxLabel.string = "Proximity: % %".format(
			sensorVal.round(0.001),
			if (sensorVal >= 0.98) { "[TOUCH]" } { "" }
		);
	});

// Momentary TOUCH button (holds sensor = 1.0 while pressed)
touchBtn = Button(win, 78@28)
	.states_([["TOUCH", Color.black, Color.grey(0.85)], ["TOUCHING", Color.white, Color.red(0.8)]])
	.mouseDownAction_({ |btn|
		touchHeld = true;
		btn.value = 1;
		proxLabel.string = "Proximity: 1.000 [TOUCH HELD]";
	})
	.mouseUpAction_({ |btn|
		touchHeld = false;
		btn.value = 0;
		proxLabel.string = "Proximity: %".format(sensorVal.round(0.001));
	});

// Sound selector
StaticText(win, 80@24).string_("Sound:");
soundMenu = PopUpMenu(win, 308@24)
	.items_(soundItems)
	.value_(0)
	.action_({ |m|
		soundId = m.value;
		addr.sendMsg('/ddsp/sound', soundId);
	});

// Note Hz slider (optional pitch override; 0 = use f0_base)
noteLabel = StaticText(win, 290@20).string_("Note Hz: base f0 (0 Hz)");
useBaseBtn = Button(win, 98@20)
	.states_([["Use Base f0"]])
	.action_({
		noteHz = 0.0;
		noteLabel.string = "Note Hz: base f0 (0 Hz)";
		addr.sendMsg('/ddsp/note', 0.0);
	});

noteSlider = Slider(win, 396@24)
	.value_(0.3)
	.action_({ |sl|
		noteHz = sl.value.linexp(0, 1, 65.0, 520.0);
		noteLabel.string = "Note Hz: % Hz".format(noteHz.round(0.1));
		addr.sendMsg('/ddsp/note', noteHz);
	});

// Gain slider + Reset button
StaticText(win, 50@24).string_("Gain:");
gainSlider = Slider(win, 240@24)
	.value_(0.5)
	.action_({ |sl|
		gain = sl.value.linlin(0, 1, 0.0, 2.0);
		addr.sendMsg('/ddsp/gain', gain);
	});

resetBtn = Button(win, 90@24)
	.states_([["Reset State"]])
	.action_({
		addr.sendMsg('/ddsp/reset');
	});

// 100 Hz OSC stream to Python
~ddspRoutine = Routine({
	inf.do {
		var val = if (touchHeld) { 1.0 } { sensorVal };
		addr.sendMsg('/ddsp/sensor', val, soundId, noteHz);
		0.01.wait; // 100 Hz
	};
}).play(SystemClock);

win.onClose = {
	if (~ddspRoutine.notNil) { ~ddspRoutine.stop; ~ddspRoutine = nil };
	"DDSP OSC sender stopped.".postln;
};

"DDSP OSC sender running -> 127.0.0.1:9000 at 100 Hz".postln;
)


/*
// ---------------------------------------------------------------------------
// OPTIONAL: Tap Sentire's live proximity signal (once Sentire is booted)
// ---------------------------------------------------------------------------
// Evaluate after Sentire's main.scd is running and g.controller exists.
(
~ddspAddr = NetAddr("127.0.0.1", 9000);
~ddspSoundId = 0;

// Replace `inBus` with Sentire's post-scaling proximity bus index
SynthDef(\sentireToDdspTap, { |inBus = 0|
	var sig = A2K.kr(In.ar(inBus, 1)).clip(0, 1);
	SendReply.kr(Impulse.kr(100), '/sentire_ddsp_val', [sig]);
}).add;

OSCdef(\sentireToDdspForward, { |msg|
	~ddspAddr.sendMsg('/ddsp/sensor', msg[3], ~ddspSoundId, 0.0);
}, '/sentire_ddsp_val');

// Example: ~tap = Synth(\sentireToDdspTap, [\inBus, g.controller.signalFlow.proximityDetection.bus.index]);
// To stop: ~tap.free; OSCdef(\sentireToDdspForward).free;
)
*/
