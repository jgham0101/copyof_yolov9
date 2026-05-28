from __future__ import annotations
from pathlib import Path
import argparse, csv, json, shutil, tarfile, xml.etree.ElementTree as ET
import requests, yaml
from tqdm import tqdm

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
VOC_URLS = {
    "trainval": [
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "https://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
    ],
    "test": [
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
        "https://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
    ],
}

def download_file(urls, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 100_000_000:
        print('exists:', dst); return
    last = None
    for url in urls:
        try:
            print('downloading:', url)
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))
                with dst.open('wb') as f, tqdm(total=total, unit='B', unit_scale=True) as pbar:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk); pbar.update(len(chunk))
            if dst.exists() and dst.stat().st_size > 100_000_000: return
        except Exception as e:
            last = repr(e); print('download failed:', url, last)
    raise RuntimeError(f'all download urls failed for {dst.name}: {last}')

def extract_tar(tar_path: Path, root: Path):
    voc = root / 'VOCdevkit' / 'VOC2007'
    if voc.exists() and (voc / 'JPEGImages').exists() and (voc / 'Annotations').exists():
        print('VOCdevkit exists:', voc); return
    print('extracting:', tar_path)
    with tarfile.open(tar_path, 'r') as tf: tf.extractall(root)

def prepare_raw(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    trainval = root / 'VOCtrainval_06-Nov-2007.tar'
    test = root / 'VOCtest_06-Nov-2007.tar'
    download_file(VOC_URLS['trainval'], trainval); extract_tar(trainval, root)
    download_file(VOC_URLS['test'], test); extract_tar(test, root)
    voc = root / 'VOCdevkit' / 'VOC2007'
    for p in ['JPEGImages', 'Annotations', 'ImageSets/Main/trainval.txt', 'ImageSets/Main/test.txt']:
        assert (voc / p).exists(), voc / p
    return voc

def read_ids(voc: Path, split: str):
    return [x.strip() for x in (voc / 'ImageSets' / 'Main' / f'{split}.txt').read_text().splitlines() if x.strip()]

def xml_to_yolo(xml_path: Path, include_difficult=False):
    root = ET.parse(xml_path).getroot()
    size = root.find('size')
    iw, ih = float(size.findtext('width')), float(size.findtext('height'))
    lines, objs = [], []
    for obj in root.findall('object'):
        name = obj.findtext('name')
        difficult = int(obj.findtext('difficult', default='0'))
        if name not in VOC_CLASSES: continue
        if difficult and not include_difficult: continue
        b = obj.find('bndbox')
        xmin = max(0.0, float(b.findtext('xmin')) - 1.0)
        ymin = max(0.0, float(b.findtext('ymin')) - 1.0)
        xmax = min(iw, float(b.findtext('xmax')) - 1.0)
        ymax = min(ih, float(b.findtext('ymax')) - 1.0)
        bw, bh = max(0.0, xmax-xmin), max(0.0, ymax-ymin)
        if bw <= 1 or bh <= 1: continue
        cx, cy = (xmin+xmax)/2/iw, (ymin+ymax)/2/ih
        nw, nh = bw/iw, bh/ih
        cls = VOC_CLASSES.index(name)
        lines.append(f'{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}')
        objs.append(name)
    return lines, objs

def build_split(voc: Path, ids, out: Path, split: str, include_difficult=False):
    img_out, lbl_out = out / 'images' / split, out / 'labels' / split
    img_out.mkdir(parents=True, exist_ok=True); lbl_out.mkdir(parents=True, exist_ok=True)
    cls_img = {c:set() for c in VOC_CLASSES}; cls_inst = {c:0 for c in VOC_CLASSES}; empty = 0
    for image_id in tqdm(ids, desc=f'convert {split}'):
        src_img, src_xml = voc / 'JPEGImages' / f'{image_id}.jpg', voc / 'Annotations' / f'{image_id}.xml'
        assert src_img.exists(), src_img; assert src_xml.exists(), src_xml
        dst_img = img_out / f'{image_id}.jpg'
        if not dst_img.exists(): shutil.copy2(src_img, dst_img)
        lines, objs = xml_to_yolo(src_xml, include_difficult)
        (lbl_out / f'{image_id}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        if not lines: empty += 1
        for c in objs:
            cls_img[c].add(image_id); cls_inst[c] += 1
    rows = [{'class_index':i, 'name':c, 'num_images':len(cls_img[c]), 'num_instances':cls_inst[c]} for i,c in enumerate(VOC_CLASSES)]
    stats = out / 'stats' / f'{split}_class_stats.csv'; stats.parent.mkdir(parents=True, exist_ok=True)
    with stats.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
    return {'split':split, 'num_images':len(ids), 'num_labels':len(list(lbl_out.glob('*.txt'))), 'empty_labels':empty, 'num_instances':sum(cls_inst.values()), 'stats_csv':str(stats)}

def validate_labels(out: Path):
    bad, total = [], 0
    for split in ['train','val']:
        for p in (out/'labels'/split).glob('*.txt'):
            txt = p.read_text(encoding='utf-8')
            if '\\n' in txt: bad.append((str(p), 'literal_backslash_n')); continue
            for i,line in enumerate(txt.splitlines(), 1):
                if not line.strip(): continue
                parts = line.split()
                if len(parts) != 5: bad.append((str(p), f'line {i}: {len(parts)}')); continue
                try:
                    cls = int(parts[0]); vals = [float(x) for x in parts[1:]]
                    if not 0 <= cls < len(VOC_CLASSES): bad.append((str(p), f'class {cls}'))
                    if not all(0.0 <= v <= 1.0 for v in vals): bad.append((str(p), f'bbox {vals}'))
                    total += 1
                except Exception as e: bad.append((str(p), repr(e)))
    if bad:
        print(bad[:20]); raise AssertionError(f'bad labels: {len(bad)}')
    print('label validation passed:', total)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--download-root', default='/content/datasets/voc_raw')
    ap.add_argument('--out-root', default='/content/datasets/voc2007_yolo')
    ap.add_argument('--yaml-out', default='data/voc2007.yaml')
    ap.add_argument('--manifest-out', default='')
    ap.add_argument('--include-difficult', action='store_true')
    args = ap.parse_args()
    voc = prepare_raw(Path(args.download_root)); out = Path(args.out_root); out.mkdir(parents=True, exist_ok=True)
    train_ids, val_ids = read_ids(voc, 'trainval'), read_ids(voc, 'test')
    train_info = build_split(voc, train_ids, out, 'train', args.include_difficult)
    val_info = build_split(voc, val_ids, out, 'val', args.include_difficult)
    validate_labels(out)
    yaml_data = {'path':str(out), 'train':'images/train', 'val':'images/val', 'nc':len(VOC_CLASSES), 'names':VOC_CLASSES}
    yaml_out = Path(args.yaml_out); yaml_out.parent.mkdir(parents=True, exist_ok=True); yaml_out.write_text(yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True), encoding='utf-8')
    manifest = {'dataset':'PASCAL VOC2007', 'include_difficult':args.include_difficult, 'classes':VOC_CLASSES, 'train':train_info, 'val':val_info, 'yaml':str(yaml_out), 'train_ids':train_ids, 'val_ids':val_ids}
    man = out / 'voc2007_manifest.json'; man.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if args.manifest_out: shutil.copy2(man, Path(args.manifest_out))
    print(json.dumps({'train_images':train_info['num_images'], 'train_instances':train_info['num_instances'], 'val_images':val_info['num_images'], 'val_instances':val_info['num_instances'], 'yaml':str(yaml_out)}, indent=2))

if __name__ == '__main__': main()
