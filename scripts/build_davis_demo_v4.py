"""Side-by-side scientific demos; eight display frames/s is not processing FPS."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/evidence/davis_v4'
OUT=ROOT/'output/research_instance_v4'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((BASE/'manifest.json').read_text());pred=json.loads((BASE/'predictions.json').read_text())
    results={(r['key'],r['method']):r for r in pred}
    methods=[('visual_baseline','Исходный визуальный'),('visual_strict','Строгий визуальный'),('gate_text_s2027_full','Текстовая проверка')]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    credit_font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
    paths=[]
    for seq in manifest['sequences']:
        if not seq['continuous']:continue
        scene=seq['sequence'];ref=next(r for r in manifest['refs'] if r['identity'].startswith(scene+':'))
        frames=sorted([r for r in manifest['frames'] if r['identity']==ref['identity']],key=lambda r:r['frame'])
        path=OUT/f'demo_{scene}.mp4';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),8,(1620,410))
        assert writer.isOpened()
        reference=Image.open(ROOT/ref['path']).convert('RGB');reference.thumbnail((170,230))
        for row in frames:
            canvas=Image.new('RGB',(1620,410),'white');draw=ImageDraw.Draw(canvas)
            draw.text((8,5),'Референс',font=font,fill='black');canvas.paste(reference,(5,40))
            draw.multiline_text((5,278),'DAVIS 2017 / 2016\nPont-Tuset et al.\nPerazzi et al.\ndavischallenge.org\nCC BY-NC 4.0\nИзменено: рамки',font=credit_font,fill='black',spacing=3)
            im=Image.open(ROOT/row['image']).convert('RGB')
            for i,(name,title) in enumerate(methods):
                x=180+i*480;panel=im.copy();d=ImageDraw.Draw(panel);r=results[(row['key'],name)]
                if row['bbox']:d.rectangle(row['bbox'],outline=(0,220,0),width=4)
                if r['bbox']:d.rectangle(r['bbox'],outline=(255,0,0),width=3)
                panel.thumbnail((476,310));canvas.paste(panel,(x,36))
                draw.text((x+4,5),title,font=font,fill='black')
                status='Отказ' if r['bbox'] is None else f'IoU {r["iou"]:.2f}'
                draw.text((x+4,350),status,font=font,fill='black')
            draw.text((180,382),f'{scene}, кадр {row["frame"]}. Зелёный — разметка, красный — предсказание. Показ 8 кадр/с.',font=font,fill='black')
            writer.write(cv2.cvtColor(np.array(canvas),cv2.COLOR_RGB2BGR))
        writer.release()
        capture=cv2.VideoCapture(str(path));ok,frame=capture.read();count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));capture.release()
        assert ok and count==len(frames)
        cv2.imwrite(str(OUT/f'demo_{scene}_preview.png'),frame)
        paths.append(dict(sequence=scene,reference_id=ref['identity'],frames=count,display_fps=8,file=path.relative_to(ROOT).as_posix(),
                          source='DAVIS 2017, Pont-Tuset et al.; DAVIS 2016, Perazzi et al.; CC BY-NC 4.0 dataset metadata'))
    (BASE/'demo_manifest.json').write_text(json.dumps(paths,ensure_ascii=False,indent=2),encoding='utf-8')
    print(paths)


if __name__=='__main__':main()
