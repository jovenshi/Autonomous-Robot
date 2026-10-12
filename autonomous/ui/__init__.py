"""Display package: the 7" screen runs in its own process (ui/screen.py).

ui/link.py is the main-process side (start, packets, commands); it never
imports pygame. ui/overlay.py and ui/panels.py draw with OpenCV onto NumPy
images, so they run and test without a display.
"""
