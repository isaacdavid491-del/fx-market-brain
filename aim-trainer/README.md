# Thumb Aim Trainer

A standalone browser-based aim trainer for touchscreen shooters. It is a
practice app: six drills, scored, with history stored on your device.

It does not interact with any game. There is no overlay, no input injection,
no screen reading, no memory access and no network calls. Nothing here can
affect a running game, and there is nothing for an anti-cheat system to find.

## Running it

Open `index.html` in a mobile browser, or serve the directory:

```
python3 -m http.server 8000 --directory aim-trainer
```

Add it to your home screen to run it fullscreen and offline.

## Drills

| Drill | Trains | Scored on |
| --- | --- | --- |
| Flick Shots | Target acquisition from a cold crosshair | Median time to target |
| Tracking | Holding aim on a strafing target | Share of time on target |
| Target Switching | Re-acquiring after a kill | Median switch time |
| Recoil Control | Pulling down through a magazine | Share of rounds on target |
| Precision Taps | Raw thumb accuracy, no crosshair | Targets per minute |
| Reaction Time | Trigger latency | Median reaction |

## Setup

Look sensitivity, aim-down-sights multiplier, fire button side, invert,
tap-to-fire, haptics and target size are all under the Setup tab.

Because every run records the sensitivity it was played at, the Progress tab
can show which sensitivity you actually perform best at once you have a few
runs at two or more settings. Change one thing at a time and give it several
runs before reading anything into it.

## Storage

Everything is kept in `localStorage` on the device. Nothing is uploaded.
"Erase all saved runs" in the Progress tab clears it.
