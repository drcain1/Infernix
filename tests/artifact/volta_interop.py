"""Python's packed v3 layouts must satisfy the C++ reader's geometry contract."""
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.artifact.schema import TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.artifact.layouts import encoded_size
with tempfile.TemporaryDirectory(prefix='infernix-volta-interop-') as tmp:
    path=Path(tmp)/'packed.infernix'
    specs=[TensorSpec('fp8',(32,128),'fp8_e4m3fn_row_bf16','volta_qpn_v1'),
           TensorSpec('nvfp4',(128,256),'nvfp4','volta_qpn_v1')]
    with ArtifactWriter(path,specs,components={'text':{'config':{}}},
                        bindings={s.id:{'object':s.id} for s in specs},max_file_bytes=8192) as writer:
        for s in specs:writer.write_object(s.id,bytes(encoded_size(s.layout,s.format,s.shape)))
    raise SystemExit(subprocess.run([sys.argv[1],str(path)]).returncode)
