# Crop-Fields-Detection
**Proof of concept** which goal is to detect crop fields using instance segmentation.
Based on Google Maps API and google maps aerial images.

**End-to-end approach** is used for easier integration into production. 
1. Retrieve region from specific google maps coordinates
2. Create grid of smaller overlaping sub regions of common input size
3. Find ROIs (polygons)
4. Convert polygons to google maps API coordinates
