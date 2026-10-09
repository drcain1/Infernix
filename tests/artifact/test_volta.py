"""Exact logical-coordinate oracle, independent of the vectorized packing algorithm."""
import struct

import pytest

from tools.artifact.codecs.volta import pack_planes, unpack_block_scales, unpack_planes
from tools.artifact.layouts import encoded_size


@pytest.mark.parametrize('n,k', [(32, 128), (128, 256), (256, 5120)])
@pytest.mark.parametrize('format', ['nvfp4', 'fp8_e4m3fn_row_bf16'])
def test_every_stored_word(n, k, format):
    nv = format == 'nvfp4'
    codes = bytes((i * 17 + i // 31) % 256 for i in range(n * k // (2 if nv else 1)))
    scales = bytes((i * 29 + i // 7) % 256 for i in range(n * (k // 16 if nv else 2)))
    packed, scale = pack_planes(codes, scales, format, (n, k))
    assert unpack_planes(packed, scale, format, (n, k)) == (codes, scales)
    assert len(packed) == len(codes) and len(scale) == len(scales)
    # Each lane's logical row is derived directly from the Volta fragment map.
    for tile in range(n // 32):
        for lane in range(32):
            row = tile * 32 + (lane // 4 % 4) * 8 + lane % 4 + lane // 16 * 4
            for col in range(k):
                if nv:
                    j = col % 8 // 2 + col % 2 * 4 + col // 8 % 2 * 8
                    index = ((tile * (k // 16) + col // 16) * 32 + lane) * 8 + j // 2
                    actual = packed[index] >> (4 * (j % 2)) & 15
                    expected = codes[(row * k + col) // 2] >> (4 * (col % 2)) & 15
                    assert actual == expected
                else:
                    index = (((tile * (k // 128) + col // 128) * 8 + col % 128 // 16) * 32 + lane) * 16 + col % 16
                    assert packed[index] == codes[row * k + col]
            if nv:
                for g in range(k // 16):
                    assert scale[(tile * (k // 16) + g) * 32 + lane] == scales[row * (k // 16) + g]
            else:
                assert scale[row * 2:row * 2 + 2] == scales[row * 2:row * 2 + 2]


def test_scale_unswizzle():
    n, k = 256, 384
    encoded = bytes((i * 13 + i // 11) % 256 for i in range(n * k // 16))
    logical = unpack_block_scales(encoded, (n, k))
    for r in range(n):
        for g in range(k // 16):
            address = (r // 128 * (k // 64) + g // 4) * 512 + r % 32 * 16 + r % 128 // 32 * 4 + g % 4
            assert logical[r * (k // 16) + g] == encoded[address]


@pytest.mark.parametrize('shape', [(31, 128), (32, 64), (0, 128)])
def test_incomplete_tiles_rejected(shape):
    with pytest.raises(ValueError):
        encoded_size('volta_qpn_v1', 'nvfp4', shape)


@pytest.mark.parametrize('format', ['nvfp4','fp8_e4m3fn_row_bf16'])
def test_converter_and_artifact_repacking_agree(tmp_path, format):
    import torch
    from tools.artifact.schema import TensorSpec, ResourceSpec
    from tools.artifact.writer import ArtifactWriter
    from tools.artifact.reader import Artifact
    from tools.artifact.tensor_output import TensorOutput
    from tools.artifact.pack_volta import pack
    n,k=256,128
    nv=format=='nvfp4'
    codes=torch.arange(n*k//(2 if nv else 1),dtype=torch.int64).remainder(126).to(torch.uint8).reshape(n,-1)
    scales=(torch.arange(n*k//16).remainder(126).to(torch.uint8).reshape(n,k//16) if nv
            else torch.full((n,),.03125,dtype=torch.bfloat16))
    divisor=struct.pack('<f',3.125) if nv else None
    components={'text':{'config':{},'resources':{'tokenizer':'tokenizer'}}}
    bindings={'text/w':{'object':'weight'},'text/first':{'parts':[{'object':'weight','range':[0,128*k]}]}}
    for name,layout in [('native','block_scale_k16_m128x4_v1' if nv else 'row_scale_v1'),('direct','volta_qpn_v1')]:
        specs=[TensorSpec('weight',(n,k),format,layout),ResourceSpec('tokenizer',4)]
        with ArtifactWriter(tmp_path/name,specs,components=components,bindings=bindings,max_file_bytes=16384) as writer:
            output=TensorOutput(writer,'weight')
            for start in (0,128):output.write_codes(start,codes[start:start+128],scales[start:start+128],divisor)
            writer.write_object('tokenizer',b'test')
    pack(tmp_path/'native',tmp_path/'packed')
    with Artifact(tmp_path/'direct') as direct,Artifact(tmp_path/'packed') as packed:
        assert direct.read_object('weight')==packed.read_object('weight')
        assert packed.directory.bindings==bindings
        assert packed.directory.components==components
        assert packed.read_object('tokenizer')==b'test'
        if nv:assert packed.read_object('weight')[-4:]==divisor
