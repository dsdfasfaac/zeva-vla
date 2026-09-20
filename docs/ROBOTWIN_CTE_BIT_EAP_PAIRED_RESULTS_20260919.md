# ZeVA CTE+BIT+EAP paired result

The frozen 10-task × 20-episode paired evaluation used identical seed,
instruction, scene, action, and video protocols for normal Base1000 and ZeVA.

- Base1000: `111/200 = 55.5%`
- ZeVA CTE+BIT+EAP: `122/200 = 61.0%`
- Paired difference: `+11/200 = +5.5pp`
- Declared engineering gate: at least `+8/200`; **PASS**
- McNemar exact `p=0.2543`
- Paired bootstrap 95% interval: `[-3,+14]pp`

All 400 videos and the full seed/instruction/result records passed the
independent audit. This is a passed engineering gate, not a statistical
significance claim. The frozen set had historical exposure, and the older
failed result Base `111/200` versus old ZeVA `106/200` remains disclosed.

The PIM extensions start from this fixed Parent. Their two training settings
are documented in [ZEVA_PIM_TRAINING_SETTINGS.md](ZEVA_PIM_TRAINING_SETTINGS.md).
