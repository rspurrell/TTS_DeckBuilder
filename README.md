# Tabletop Simulator Deck Builder

Detects, separates, and straightens multiple scanned objects from a single image. Then combine all output images into a single image grid for use in Tabletop Simulator.

## Install Required Packages
`pip install opencv-python numpy pillow`

## Parse Your Scans
`python deck_builder.py scan_0001.jpg . --orientation landscape --final-crop 21,21,2042x1445 --resize-to 578x409 --gamma 0`

## Build Your Deck
Requires that all images have the same dimensions.

`python deck_builder.py . . --grid 10x3`
