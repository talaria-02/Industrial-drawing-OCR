"""
적층 비대칭공차 rec 데이터 생성 (검수용 프리뷰 20세트)
=========================================================
"2,5 (위:+0,2 아래:0)" 같은 적층 공차 표기를 대상으로 함.

설계 결정: 위/아래를 한 crop에 합치지 않음 — rec은 CTC 기반
한 줄 인식기라 2줄 crop을 애초에 못 품. 공칭값/상한/하한을
각각 별도의 작은 crop으로 만들어 실제 검출기 박스 분리와 맞춤.
상한·하한은 공칭값보다 작은 폰트로 렌더링(실제 도면 관례 반영).

출력:
  finetune/stacked_tol_preview/preview.png   (검수용 대조표)
  finetune/stacked_tol_preview/images/*.png  (개별 crop)
  finetune/stacked_tol_preview/rec_gt.txt
"""

import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

random.seed(42)

OUT_DIR = Path('finetune/stacked_tol_preview')
IMG_DIR = OUT_DIR / 'images'
FONT_PATH = r'C:\Windows\Fonts\arial.ttf'


def get_font(size):
    return ImageFont.truetype(FONT_PATH, size)


def rand_nominal():
    whole = random.randint(1, 200)
    if random.random() < 0.4:
        frac = random.choice(['5', '25', '75'])
        sep = random.choice(['.', ','])
        return f'{whole}{sep}{frac}'
    return str(whole)


def rand_tol_value():
    """상한/하한 값. 0인 경우도 흔함(한쪽이 공차 없는 비대칭 공차)."""
    if random.random() < 0.3:
        return '0'
    frac = random.choice(['05', '1', '2', '15'])
    sep = random.choice(['.', ','])
    return f'0{sep}{frac}'


def gen_set():
    """(nominal, upper, lower) 텍스트 세트 하나 생성.
    0이 아닌 공차값엔 반드시 부호가 붙어야 함(방향 표시라 부호 없으면 의미 불명)."""
    nominal = rand_nominal()
    upper = f'+{rand_tol_value()}'
    if upper == '+0':
        upper = random.choice(['+0', '0'])  # 상한 0은 부호 있어도/없어도 실무에서 둘 다 봄

    lv = rand_tol_value()
    if lv == '0':
        lower = '0'   # 0인 경우 "-0" 표기는 실무에서 거의 안 씀
    else:
        lower = f'-{lv}'  # 하한은 관례상 거의 항상 음수 부호
    return nominal, upper, lower


def render_crop(text, size, pad=8):
    font = get_font(size)
    tmp = Image.new('RGB', (10, 10), (255, 255, 255))
    bbox = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + pad * 2
    h = bbox[3] - bbox[1] + pad * 2
    img = Image.new('RGB', (max(w, 10), max(h, 10)), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=(0, 0, 0))
    return img


def main(n_sets=20):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    labels = []
    rows = []  # 프리뷰용: (nominal_img, upper_img, lower_img, texts)

    for i in range(n_sets):
        nominal, upper, lower = gen_set()
        nom_size = random.randint(32, 42)
        tol_size = int(nom_size * random.uniform(0.5, 0.65))  # 상/하한 = 공칭값의 절반~2/3 크기

        nom_img = render_crop(nominal, nom_size)
        up_img = render_crop(upper, tol_size)
        lo_img = render_crop(lower, tol_size)

        for suffix, img, text in [('nom', nom_img, nominal),
                                  ('up', up_img, upper),
                                  ('lo', lo_img, lower)]:
            name = f'set{i:02d}_{suffix}.png'
            img.save(IMG_DIR / name)
            labels.append(f'{IMG_DIR / name}\t{text}')

        rows.append((nom_img, up_img, lo_img, nominal, upper, lower))

    with open(OUT_DIR / 'rec_gt.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels) + '\n')

    # 검수용 대조표 조립
    row_h = 90
    col_w = 500
    preview = Image.new('RGB', (col_w * 2, row_h * n_sets), (255, 255, 255))
    pd = ImageDraw.Draw(preview)
    label_font = get_font(16)

    for i, (nom_img, up_img, lo_img, nominal, upper, lower) in enumerate(rows):
        y = i * row_h
        # 왼쪽: 실제 렌더링된 3개 crop 나열
        x = 10
        preview.paste(nom_img, (x, y + 20))
        x += nom_img.width + 15
        # 상/하한은 위아래로 살짝 겹쳐 보이게 배치(실제 배치 흉내)
        preview.paste(up_img, (x, y + 5))
        preview.paste(lo_img, (x, y + 5 + up_img.height + 4))
        # 오른쪽: 정답 텍스트 표기
        pd.text((col_w + 10, y + 35), f'#{i:02d}  nominal={nominal!r}  upper={upper!r}  lower={lower!r}',
                font=label_font, fill=(0, 0, 150))
        pd.line([(0, y + row_h), (preview.width, y + row_h)], fill=(220, 220, 220))

    preview.save(OUT_DIR / 'preview.png')
    print(f'생성 완료: {n_sets}세트 ({n_sets*3}장) -> {IMG_DIR}')
    print(f'검수용 대조표: {OUT_DIR / "preview.png"}')


if __name__ == '__main__':
    main()
