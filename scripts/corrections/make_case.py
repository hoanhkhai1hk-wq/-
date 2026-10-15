"""用固定公开样例生成明确人工注入异常的案例；不改原样例，不运行模型。"""
import hashlib,json,sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import tifffile
from recognition.raster_io import read_image,_extratag_for
SOURCE=ROOT/'samples/ftw_smoke/inference-img.tif'
EXPECTED='da482e7987b36898bd9f8381e7c7fe95c05dfc5ccbba19a58eb65e544d1af750'
original=SOURCE.read_bytes();assert hashlib.sha256(original).hexdigest()==EXPECTED
out=ROOT/'reports/corrections/cases'/datetime.now().strftime('case-%Y%m%d-%H%M%S-%f')
out.mkdir(parents=True,exist_ok=False)
image,valid,meta=read_image(SOURCE)
assert valid.all() and image.shape==(379,578,8)
changed=image.copy();changed[100:116,200:216,:]=np.nan
target=out/'ARTIFICIAL-invalid-256-pixels.tif'
tags=[_extratag_for(int(code),value) for code,value in meta['geotiff_tags'].items()]
tifffile.imwrite(target,changed,photometric='minisblack',planarconfig='contig',metadata=None,
                  description='ARTIFICIAL TEST: 16x16 invalid pixels injected; not field observations or ground truth.',
                  software='AgroLake correction workflow test',extratags=tags)
back,back_valid,back_meta=read_image(target)
assert int(np.count_nonzero(~back_valid))==256
assert np.array_equal(back[back_valid],image[back_valid])
assert back_meta['geotiff_tags']==meta['geotiff_tags']
assert SOURCE.read_bytes()==original
result={'directory':str(out),'case_kind':'artificial_invalid_pixel_injection',
        'source_sha256':EXPECTED,'source_bytes':len(original),
        'problem_file':target.name,'problem_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'shape_hw':[379,578],'bands':8,'total_pixels':379*578,'injected_invalid_pixels':256,
        'block_half_open':{'rows':[100,116],'cols':[200,216]},'outside_block_values_equal':True,
        'spatial_tags_preserved':True,'original_bytes_preserved':True,
        'repair_method':'Restore input from the pinned clean public reference. This is a controlled test, not recovery of unknown observations.',
        'boundary':'Artificial data-integrity case; not segmentation ground truth, accuracy, measured field area, or natural business evidence.'}
(out/'case.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
