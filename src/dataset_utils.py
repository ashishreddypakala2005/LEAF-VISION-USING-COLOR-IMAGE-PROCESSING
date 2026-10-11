from pathlib import Path
EXTENSIONS={".jpg",".jpeg",".png",".bmp",".webp"}

def list_dataset_images(root, class_name=None):
    root=Path(root);out=[]
    if not root.exists(): return out
    folders = [root / class_name] if class_name else sorted(p for p in root.iterdir() if p.is_dir())
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                out.append({"class":folder.name,"path":path,"relative":str(path.relative_to(root))})
    return out

def scan_dataset(root):
    root=Path(root)
    if not root.exists() or not root.is_dir(): return {"image_count":0,"class_count":0,"size_mb":0.0,"classes":[]}
    rows=[];total=0;bytes_=0
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        pics=[p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS]
        if pics:
            rows.append((folder.name,len(pics)));total+=len(pics)
            for p in pics:
                try:bytes_+=p.stat().st_size
                except OSError:pass
    loose=[p for p in root.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]
    if loose:
        rows.append(("Unsorted (root)",len(loose)));total+=len(loose)
        for p in loose:
            try:bytes_+=p.stat().st_size
            except OSError:pass
    return {"image_count":total,"class_count":len(rows),"size_mb":bytes_/1048576,"classes":rows}
