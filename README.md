# Tabletop Simulator Deck Builder

Detects, separates, and straightens multiple scanned objects from a single image. Then combine all output images into a single image grid for use in Tabletop Simulator.

## Install Required Packages
`pip install opencv-python numpy pillow`

## Parse Your Scans
`python deck_builder.py scan_0001.jpg output --orientation landscape --final-crop 21,21,2042x1445 --resize-to 578x409 --gamma 0`

| Option | Description |
| ------ | ----------- |
| `--min-area` | Minimum area (in pixels) for a detected contour to be considered valid (avoids artifacts) |
| `--adaptive-block-size` | Block size for adaptive thresholding (must be odd and > 1) |
| `--adaptive-threshold` | Adjusts adaptive threshold sensitivity. Higher values make the detected image cleaner (reduces noise), but might lose fine detail. Lower values can preserve smaller or faint objects, but may introduce noise or false contours. |
| `--orientation` | Enforce preferred orientation: landscape or portrait |
| `--final-crop` | Crop final image to a region: 'x,y,WIDTHxHEIGHT' (e.g. '19,21,2042x1445') |
| `--resize-to` | resize output image to WIDTHxHEIGHT using gamma correction and LANCZOS resampling |
| `--gamma` | Gamma correction value when resizing (0 to disable, default=2.2). |
| `--debug` | Show debug previews and allow image confirmation. Enter to save. Space to skip. +/-/z to zoom. Use with adaptive block-size and threshold to fine tune the detected contour |

## Build Your Deck
Combine all output images into a single image grid. Use option with format COLSxROWS (e.g. 6x3). All images should have the same dimensions.

`python deck_builder.py ./source_folder ./output_folder --grid 10x3`
