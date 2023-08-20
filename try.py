import clip


net = clip.load("ViT-B/32")[0]
print(net.visual.input_resolution)