from layers import PixelEncDec


def img2pattern_model(inp_channels=3, out_channels=3, dim=32, num_blocks=[4, 6, 6, 8], 
                      heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=True):
    model = PixelEncDec(inp_channels=inp_channels,
                        out_channels=out_channels,
                        dim=dim,
                        num_blocks=num_blocks,
                        heads=heads,
                        ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias)
    return model

def pattern2img_model(inp_channels=3, out_channels=3, dim=32, num_blocks=[4, 6, 6, 8],
                      heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=True):
    model = PixelEncDec(inp_channels=inp_channels,
                        out_channels=out_channels,
                        dim=dim,
                        num_blocks=num_blocks,
                        heads=heads,
                        ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias)
    return model