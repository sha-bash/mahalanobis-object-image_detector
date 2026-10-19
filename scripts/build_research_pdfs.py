"""Build a complete Russian report and an exactly two-page scientific manuscript.

Run with the bundled document runtime. All result tables read saved experiment JSON.
"""
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER,TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle,getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image
from PIL import Image as PILImage
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/evidence/multimodal_v1'
OUT=ROOT/'output/research_20261010'
for name,file in [('TimesRU','times.ttf'),('TimesRU-Bold','timesbd.ttf'),('TimesRU-Italic','timesi.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
pdfmetrics.registerFontFamily('TimesRU',normal='TimesRU',bold='TimesRU-Bold',italic='TimesRU-Italic',boldItalic='TimesRU-Bold')
ST=getSampleStyleSheet()
ST.add(ParagraphStyle('BodyRU',fontName='TimesRU',fontSize=10.7,leading=13.4,spaceAfter=7,alignment=TA_JUSTIFY))
ST.add(ParagraphStyle('TitleRU',fontName='TimesRU-Bold',fontSize=18,leading=21,spaceAfter=13))
ST.add(ParagraphStyle('HeadRU',fontName='TimesRU-Bold',fontSize=13,leading=16,spaceBefore=7,spaceAfter=8,keepWithNext=True))
ST.add(ParagraphStyle('SmallRU',fontName='TimesRU',fontSize=8.6,leading=10.8,spaceAfter=5))
ST.add(ParagraphStyle('CellRU',fontName='TimesRU',fontSize=8.2,leading=10))
ST.add(ParagraphStyle('PaperBody',fontName='TimesRU',fontSize=10.4,leading=12.3,spaceAfter=5,alignment=TA_JUSTIFY))
ST.add(ParagraphStyle('PaperHead',fontName='TimesRU-Bold',fontSize=10.8,leading=13,spaceBefore=6,spaceAfter=5,keepWithNext=True))
ST.add(ParagraphStyle('PaperTitle',fontName='TimesRU-Bold',fontSize=14,leading=16,spaceAfter=8,alignment=TA_CENTER))
ST.add(ParagraphStyle('CenterRU',fontName='TimesRU',fontSize=10,leading=12,spaceAfter=4,alignment=TA_CENTER))


def para(text,style='BodyRU'):
    return Paragraph(text,ST[style])


def table(rows,widths=None):
    cells=[[para(str(v),'CellRU') for v in row] for row in rows]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8edf1')),
                          ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#526473')),
                          ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d8d8d8')),
                          ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),4),
                          ('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),5)]))
    return t


def picture(path,width=490,max_height=250):
    with PILImage.open(path) as image:
        w,h=image.size
    scale=min(width/w,max_height/h)
    return Image(str(path),width=w*scale,height=h*scale,hAlign='CENTER')


def footer(canvas,doc):
    canvas.setFont('TimesRU',8)
    canvas.setFillColor(colors.HexColor('#555555'))
    canvas.drawString(17*mm,11*mm,'Исследование поиска экземпляров · 10 октября 2026')
    canvas.drawRightString(A4[0]-17*mm,11*mm,str(doc.page))


def label(method):
    if method=='sift': return 'SIFT'
    modality,kind=method.split('_',1)
    return {'visual':'V','text':'T','fusion':'V+T'}[modality]+' / '+{'raw_cos':'cos','pca_cos':'PCA cos','diag':'PCA diag M','full':'PCA full M'}[kind]


def pct(value): return f'{100*value:.1f}'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result=json.loads((BASE/'results.json').read_text())
    captions=json.loads((BASE/'captions.json').read_text(encoding='utf-8'))
    timing=json.loads((BASE/'feature_provenance.json').read_text())
    collisions=json.loads((BASE/'caption_collisions.json').read_text(encoding='utf-8'))
    macro=result['macro']
    def stat(method,k,scope='real'):
        return next(r for r in macro if r['method']==method and r['k']==k and r['scope']==scope)
    comparisons=[]
    for k in [1,3]:
        for a,b in [('visual_raw_cos','fusion_raw_cos'),('fusion_pca_cos','fusion_full')]:
            x,y=stat(a,k),stat(b,k)
            comparisons.append(f'K={k}, {label(a)} → {label(b)}: TPR {pct(x["tpr"])} → {pct(y["tpr"])}%; FPR {pct(x["fpr"])} → {pct(y["fpr"])}%; BA {pct(x["balanced_accuracy"])} → {pct(y["balanced_accuracy"])}%.')
    report=[]
    report.append(para('Визуальная и текстовая информация в поиске экземпляра объекта','TitleRU'))
    report.append(para('Полный отчёт о плане, выполненных экспериментах и границах выводов','HeadRU'))
    report.append(para('Исследуется поиск конкретного экземпляра автомобиля по одному или трём изображениям. Проверены классическое сопоставление SIFT, визуальные признаки DINOv2, текстовые описания VLM и их совместное использование. Основной новый эксперимент отделяет распознавание экземпляра от генерации рамок: все методы получают одни и те же размеченные области.'))
    report.append(para('Объект исследования — поиск и локализация заданного экземпляра на изображениях и видео. Предмет — способы объединения визуальных и текстовых признаков, метрика сходства и правило отказа. Цель — установить, уменьшается ли принятие посторонних экземпляров при сохранении приемлемой полноты.'))
    report.append(para('<b>Основной вывод.</b> Устойчивый полезный выигрыш объединения не установлен. При трёх референсах raw fusion повышает BA относительно visual cosine, но TPR остаётся ниже 70%. Полная ковариация совместных PCA-признаков не улучшила BA относительно косинуса на тех же признаках. Это отрицательный результат конкретной реализации на малой выборке, а не опровержение класса мультимодальных методов.'))
    report.append(para('Главные измеренные сравнения','HeadRU'))
    for text in comparisons: report.append(para(text))
    report.append(para('Эти числа относятся к классификации пар готовых GT-вырезок. Они не являются новой оценкой детекции или сопровождения. Для всей системы используются отдельные, ранее сохранённые видеопрогоны; уровни оценки не объединяются.'))
    report.append(para('Объём и разбиение','HeadRU'))
    report.append(table([['Материал','Экземпляры','Изображения','Статус'],['car-0, car-1, car-2','18','90','Три реальные сцены'],['car-3','6','30','Визуально игровая сцена'],['Всего','24','120','Три референса и два кандидата на ID']], [175,70,80,170]))
    report.append(para('Источник — официальный GMOT-40 v0.1, Bai и соавт., CVPR 2021; CC BY-NC-SA 4.0. Ролики уже участвовали в предварительном исследовании. Новый анализ имеет исследовательский статус, а не статус нетронутого подтверждающего теста. Пересечение с предобучением энкодеров неизвестно.','SmallRU'))
    report.append(PageBreak())
    report.append(para('План и воспроизводимый протокол','TitleRU'))
    report.append(para('План зафиксирован коммитом 81a900e до извлечения новых признаков. Гипотеза H1: текстовые признаки дополняют визуальные и уменьшают ложные принятия при TPR не ниже 0,70. Гипотеза H2: регуляризованная обученная метрика улучшает сравнение относительно косинуса в том же пространстве признаков.'))
    report.append(table([['Fold','Обучение','Валидация','Оценка'],['1','car-0','car-1','car-2'],['2','car-1','car-2','car-0'],['3','car-2','car-0','car-1'],['Игровой контроль','car-0','car-1','car-3']],[125,120,125,125]))
    report.append(para('Отбор не зависит от результатов модели: первые шесть ID по номеру, имеющие не менее 61 размеченного кадра с рамкой от 40×20 пикселей, source frame &lt;300. Позиции 0, 10 и 20 в списке пригодных кадров задают референсы; позиции 40 и 60 — кандидаты. Это разнесённые по времени кадры одной съёмки, а не независимые съёмки одного экземпляра.'))
    report.append(para('Каждая сцена формирует 6 запросов × 12 кандидатов = 72 пары: 12 положительных и 60 отрицательных. На трёх реальных сценах — 216 пар на метод и протокол K, но только 18 целевых ID в трёх сценах. Два кандидата одного экземпляра и пары с общим референсом зависимы.'))
    report.append(para('Для K=1 используется первый референс; для K=3 — все три. Расстояние до кандидата — минимум по референсам. Для SIFT берётся максимум доли хороших соответствий (Lowe ratio 0,75). Отсутствие дескрипторов даёт нулевое сходство и не удаляет пример.'))
    report.append(para('Порог выбирается только на валидационной сцене: минимальная FPR при TPR≥0,70; далее максимальная TPR и минимальный порог. Равные оценки принимаются одной группой. PCA и ковариация обучаются только на тренировочной сцене; всё пересчитывается для каждого fold.'))
    report.append(para('Top1 — доля запросов, для которых ближайший из 12 кандидатов принадлежит цели. При равенстве минимальных оценок начисляется доля правильных среди связанных кандидатов; это устраняет преимущество порядка ID. В галерее всегда два правильных кандидата. Top1 не оценивает отказ при отсутствующей цели.'))
    report.append(para('Выполненные этапы: подготовка фиксированного manifest; новые независимые VLM-описания; извлечение DINO/SBERT; три реальных fold и отдельный игровой контроль; 13 методов × 2 значения K × 4 fold = 104 конфигурации оценки. Нейросетевые веса не дообучались.'))
    report.append(PageBreak())
    report.append(para('Признаки и обученная метрика','TitleRU'))
    report.append(para('DINOv2-small даёт визуальный CLS-вектор, Qwen3.5-0.8B независимо описывает каждую вырезку, all-MiniLM-L6-v2 преобразует описание в текстовый вектор. Оба вектора нормируются. Для объединения используются равные фиксированные веса; текст кандидата не получает описание эталона.'))
    report.append(picture(BASE/'equation_1.png',465,40))
    report.append(para('В каждой модальности есть raw cosine baseline. Для контролируемого сравнения метрик PCA обучается на 30 тренировочных вырезках и сокращает размерность до восьми. После проекции векторы снова нормируются.'))
    report.append(picture(BASE/'equation_2.png',430,43))
    report.append(para('Ковариация оценивается по разностям признаков одного ID: три тренировочных референса × два кандидата × шесть ID = 36 разностей. Они симметризуются знаками ±. Полная ковариация использует Ledoit–Wolf shrinkage; диагональный вариант усредняет дисперсии с общей средней с коэффициентом 0,5. В обоих вариантах добавляется ridge 10⁻⁴.'))
    report.append(picture(BASE/'equation_3.png',450,42))
    report.append(picture(BASE/'equation_4.png',475,43))
    report.append(para('PCA + cosine, PCA + diagonal M и PCA + full M используют одинаковые проекции. Поэтому разность между ними относится к метрике. Разность raw cosine и PCA cosine показывает влияние самой проекции. Совместная PCA и полная матрица допускают взаимодействие модальностей, но не равнозначны надёжной оценке полной ковариации исходных 768 признаков.'))
    report.append(picture(BASE/'equation_5.png',475,40))
    report.append(para('TPR — доля принятых правильных пар, FPR — доля принятых чужих пар, BA — сбалансированная точность. Precision зависит от состава галереи и здесь соответствует соотношению положительных и отрицательных пар 1:5. Ни одна из этих метрик сама по себе не равна успешности локализации на видео.'))
    report.append(picture(BASE/'equation_6.png',430,38))
    report.append(para('Последняя формула объясняет ограничение старого изотропного варианта: он эквивалентен косинусному порогу. В новом опыте проверяется именно оценённая ковариация, а не этот частный случай.'))
    report.append(PageBreak())
    report.append(para('Результаты по трём реальным сценам','TitleRU'))
    report.append(para('Равновесное среднее по трём сценам. Все значения в процентах; K — число референсов. Пороги взяты с других валидационных сцен. V — DINOv2, T — текст VLM/SBERT, M — Махаланобис. Никакой метод не выбирался по этим test-значениям.','SmallRU'))
    rs=[r for r in macro if r['scope']=='real']
    report.append(table([['Метод','K','TPR','FPR','BA','Top1']]+[[label(r['method']),r['k'],pct(r['tpr']),pct(r['fpr']),pct(r['balanced_accuracy']),pct(r['top1'])] for r in rs],[175,28,70,70,75,77]))
    report.append(para('Возможный рост BA не означает пригодности рабочего порога: необходимо отдельно проверить TPR≥70% и ложные принятия на каждой сцене. Все исходные TP/FP/FN/TN и пороги приведены в приложении и results.csv.','SmallRU'))
    report.append(PageBreak())
    report.append(para('Графическое сравнение','TitleRU'))
    report.append(picture(BASE/'balanced_accuracy.png',500,230))
    report.append(para('Рисунок 1. Влияние модальности, проекции и метрики. Пунктир BA=0,5 соответствует уровню решения, принимающего все или отвергающего все пары. Высота столбцов — среднее трёх сцен, не доверительный интервал.','SmallRU'))
    report.append(picture(BASE/'operating_points.png',500,230))
    report.append(para('Рисунок 2. Рабочие точки: малые круги — сцены, кресты — средние. Желательно движение влево при сохранении высокой TPR. Разброс между сценами важнее различий в сотых долях на зависимых парах.','SmallRU'))
    report.append(para('Обобщение на независимую съёмку эталона здесь не проверялось. Даже когда обучение и оценка разведены по сценам, внутри тестового запроса эталон и кандидат происходят из одного видео.'))
    report.append(para('На снежной car-2 при K=1 raw fusion снизил FPR с 96,7 до 75,0% при TPR=100%; на ночной car-1 FPR выросла с 55,0 до 81,7%. На car-0 перенесённый порог отверг все правильные пары. Эти разнонаправленные эффекты объясняют, почему один средний показатель недостаточен.'))
    report.append(PageBreak())
    report.append(para('Связь с полным поиском на видео','TitleRU'))
    report.append(para('Ниже приведены ранее выполненные online-абляции. Их нельзя объединять с новым парным экспериментом как один benchmark: различаются единицы подсчёта, объекты и генерация кандидатов.'))
    report.append(table([['Видео и цель','Геометрия','Геометрия + текст','Пояснение'],['Исходная ГАЗель','391/391','391/391','mIoU 0,811; та же сцена'],['Исходный объект 66','402/722','0/722','4 правильных кандидата отклонены'],['GMOT car-0','34/227','34/227','Ночная дорога'],['GMOT car-1','64/89','64/89','210 кадров по цели не оценены'],['GMOT car-2','Ошибка','Ошибка','Недостаточно SIFT-точек'],['GMOT car-3','165/165','165/165','Игровая сцена']],[150,80,95,170]))
    report.append(para('Для трёх завершённых GMOT парных прогонов все 691 решения и рамки совпали. В каждом семантическом прогоне выполнена одна проверка, принятая модулем. Геометрия не выдала рамок, соответствующих чужим размеченным ID: уменьшение таких ошибок данным online опытом не измеряется.'))
    report.append(para('Суммарное runner-время завершённых пар составило 53,31 с без семантики и 198,60 с с ней. Это один CPU-запуск, включая подготовку моделей. В предыдущей панели готовых кандидатов реальных сцен текст принял 6/6 правильных вырезок и отклонил 3/6 чужих; этот локальный эффект не изменил online результат.'))
    report.append(para('Исторические показатели исходной статьи — 722 кадра, 665 выданных рамок, три принятые проверки, 737,47 с — подтверждены пересчётом артефактов. Число выданных рамок не является полнотой. Исходная разметка получена трекером и проверена человеком по уточнению владельца.'))
    report.append(para('Новый эксперимент расширяет именно проверку представлений и обученной метрики. Интеграция выбранной после независимой валидации мультимодальной модели в длительное сопровождение с выходом и повторным появлением остаётся следующим этапом.'))
    report.append(PageBreak())
    report.append(para('Затраты, ошибки и границы выводов','TitleRU'))
    report.append(table([['Этап нового опыта','Объём','Измеренное время, с'],['VLM-описания',str(len(captions['records'])),f"{captions['wall_sec']:.2f}"],['DINOv2-векторы','120',f"{timing['visual_sec']:.2f}"],['SBERT-векторы','120',f"{timing['text_sec']:.2f}"],['SIFT-матрица','120 × 120',f"{timing['sift_pair_matrix_sec']:.2f}"],['PCA, ковариации и оценка','104 конфигурации',f"{result['wall_sec']:.2f}"]],[230,110,155]))
    report.append(para('CPU Intel Core i5-1240P, около 8 ГБ RAM; лимит вычислительных потоков энкодеров — четыре. Описания вычислялись отдельно от DINO/SBERT для ограничения памяти. Значения включают соответствующую загрузку модели; во время разработки могла выполняться вспомогательная работа. Это журнал затрат, а не многократно повторённый benchmark скорости.'))
    report.append(para(f'В новых описаниях найдено {len(collisions)} групп одинакового текста без учёта регистра, относящихся к разным ID. Совпадение текста может делать разные экземпляры неразличимыми для текстовой метрики. При равных эмбеддингах ни обученная ковариация, ни новый порог не восстанавливают утраченную информацию.'))
    if collisions:
        example=collisions[0]
        report.append(para('Пример: '+escape(example['caption'])+'; ID: '+escape(', '.join(example['identities']))+'.','SmallRU'))
    report.append(para('Ограничения: шесть ID на сцену; три реальные сцены; предварительная известность набора; отсутствие проверенных отрицательных видеоинтервалов; GT-кандидаты без ошибок детектора; неизвестное пересечение с предобучением. Нельзя считать 216 пар независимыми испытаниями. Проверка нескольких метрик увеличивает риск случайного «лучшего» результата. Поэтому отсутствуют p-value и подтверждающие интервалы.'))
    report.append(para('Следующий подтверждающий этап: не менее 24 реальных экземпляров в независимых съёмках, заранее выделенные train/validation/test по связанным сценам и ID; отдельные эталоны из другой съёмки; похожие машины, перекрытие, отсутствие и возвращение. Разметку отрицательных интервалов проверять вручную. Критерий выбора и размер исследования фиксировать до просмотра test.'))
    report.append(para('Утверждать можно результаты перечисленных конфигураций на выбранных данных и фактические ошибки. Пока нельзя утверждать универсальное превосходство текста или Махаланобиса, перенос на независимую съёмку, надёжность при отсутствующей цели или улучшение полноценного трекера.'))
    report.append(PageBreak())
    report.append(para('Воспроизведение и источники','TitleRU'))
    report.append(para('Исходники, выборка и журналы находятся в исследовательской ветке codex/itnt-2027-audit. Протокол: experiments/MULTIMODAL_PROTOCOL.md. Артефакты: experiments/evidence/multimodal_v1. Изображения и тяжёлые веса остаются вне Git. Таблицы и графики строятся из сохранённых оценок, а не вводятся вручную.'))
    for command in ['.venv/Scripts/python.exe scripts/multimodal_data.py describe','.venv/Scripts/python.exe scripts/multimodal_data.py features','.venv/Scripts/python.exe scripts/evaluate_multimodal.py','.audit-venv/Scripts/python.exe scripts/plot_multimodal.py','.venv/Scripts/python.exe -m pytest -q']:
        report.append(para(escape(command),'SmallRU'))
    report.append(para('Первичную подготовку выполняет стадия prepare; она отказывается перезаписывать каталог. Для полного повторения нужен новый каталог данных и независимый журнал. Стадия describe умеет продолжить прерванный запуск; попадания в кэш отмечаются. Подробные команды — MULTIMODAL_REPRODUCE.md.'))
    report.append(para('Ревизии моделей: DINOv2-small ed25f3a31f01632728cabb09d1542f84ab7b0056; SBERT 1110a243fdf4706b3f48f1d95db1a4f5529b4d41. Теги и digest VLM, параметры генерации, версии библиотек и SHA256 входов сохранены в provenance.'))
    report.append(para('Проверки включают неизменность PCA/ковариации при изменении тестовых векторов, запрет тренировочных пар из другой части, корректность порога при равных score и эквивалентность изотропного Махаланобиса косинусу. Конкретный итог тестов записывается в verification.json после завершения работы.'))
    references=[
        'Bai H. et al. GMOT-40: A Benchmark for Generic Multiple Object Tracking. CVPR, 2021, pp. 6719–6728. https://spritea.github.io/GMOT40/',
        'Oquab M. et al. DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193, 2023. https://github.com/facebookresearch/dinov2',
        'Reimers N., Gurevych I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP, 2019. https://aclanthology.org/D19-1410/',
        'Ledoit O., Wolf M. A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis, 2004, 88(2), 365–411. Реализация sklearn.covariance.LedoitWolf.',
        'Lowe D. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision, 2004, 60, 91–110.',
        'ИТНТ-2027. Правила участия и материалы: https://itnt-conf.org/poryadok-uchastiya ; https://itnt-conf.org/instruktsii . Проверено 10.10.2026.'
    ]
    for i,text in enumerate(references,1): report.append(para(f'[{i}] '+escape(text),'SmallRU'))
    report.append(para('Сроки на официальных страницах расходятся: главная — 13.11.2026/25.12.2026, отдельная страница дат — 31.10.2026/14.12.2026. Практический план сохраняет ранние даты до уточнения. PDF тезисов проверен на две страницы; соответствие бинарному Word-шаблону конференции требует отдельной проверки верстки.','SmallRU'))
    for clip in ['car-2','car-0','car-1','car-3']:
        report.append(PageBreak())
        report.append(para('Приложение Результаты сцены '+clip,'TitleRU'))
        rs=[r for r in result['rows'] if r['test']==clip]
        report.append(para(f'Обучение: {rs[0]["train"]}; валидация: {rs[0]["validation"]}. 12 положительных и 60 отрицательных пар на конфигурацию. Все пороги получены на валидации. Игровая сцена car-3 не включена в основные средние.','SmallRU'))
        report.append(table([['Метод','K','Порог','TP/12','FP/60','BA, %','Top1, %']]+[[label(r['method']),r['k'],f'{r["threshold"]:.4f}',r['tp'],r['fp'],pct(r['balanced_accuracy']),pct(r['top1'])] for r in rs],[158,24,70,57,57,63,66]))
    report_path=OUT/'research_report.pdf'
    SimpleDocTemplate(str(report_path),pagesize=A4,rightMargin=17*mm,leftMargin=17*mm,topMargin=17*mm,bottomMargin=19*mm,title='Визуальная и текстовая информация в поиске экземпляра объекта',author='В. О. Шабашев').build(report,onFirstPage=footer,onLaterPages=footer)

    paper=[]
    def pp(text): return para(text,'PaperBody')
    def ph(text): return para(text,'PaperHead')
    paper.append(para('Визуальные и текстовые признаки в поиске экземпляра объекта по ограниченному числу изображений','PaperTitle'))
    paper.append(para('В. О. Шабашев','CenterRU'))
    paper.append(para('Ульяновский государственный технический университет, Ульяновск, Россия','CenterRU'))
    paper.append(pp('<b>Аннотация.</b> Сравниваются визуальные признаки DINOv2, описания VLM/SBERT и их объединение для поиска экземпляров автомобилей. На GMOT-40 параметры и пороги определяются без использования оцениваемой сцены. При одном референсе объединение снизило сбалансированную точность с 58,1 до 55,8%; при трёх повысило с 56,7 до 59,4%, но полнота осталась ниже 70%. Обученная полная ковариация совместных PCA-признаков устойчивого выигрыша не дала.'))
    paper.append(pp('<b>Ключевые слова:</b> поиск экземпляра; мультимодальные признаки; Sentence-BERT; расстояние Махаланобиса; визуальное сопоставление.'))
    paper.append(ph('1 Введение и постановка задачи'))
    paper.append(pp('Поиск заданного экземпляра отличается от детекции категории: сходство цвета и типа автомобиля недостаточно для установления идентичности. Проверяется гипотеза, что текстовое описание дополняет визуальные признаки и уменьшает принятие посторонних объектов при приемлемой полноте. Отдельная гипотеза касается пользы обученной метрики Махаланобиса. Их разделение позволяет отличить вклад информации от вклада метрики.'))
    paper.append(pp('Вход — K референсных изображений и изображение-кандидат. В полном поиске кандидаты порождаются геометрией или детектором. Чтобы исключить этот источник различий, новый опыт использует одинаковые GT-вырезки для всех методов. Это проверка распознавания экземпляра; успешность локализации на видео оценивается отдельно.'))
    paper.append(ph('2 Представление и метрика'))
    paper.append(pp('Нормированные визуальный вектор DINOv2-small [2] и текстовый вектор all-MiniLM-L6-v2 [3] объединяются с равными весами. Тексты получены независимыми вызовами Qwen3.5-0.8B без передачи описания эталона кандидату.'))
    paper.append(picture(BASE/'equation_1.png',460,31))
    paper.append(pp('PCA до восьми компонент обучается на тренировочной сцене; проекции повторно нормируются. Ковариация оценивается по симметризованным разностям признаков одного экземпляра, с регуляризацией Ledoit–Wolf [4] и добавлением 10⁻⁴I. Проверяются также диагональная ковариация и косинус на тех же проекциях.'))
    paper.append(picture(BASE/'equation_4.png',465,34))
    paper.append(pp('Здесь u — нормированная PCA-проекция, r<sub>j</sub> — референс. Для K=3 берётся минимальное расстояние. Изотропный вариант Σ=0,01I для нормированных векторов эквивалентен косинусному критерию и не считается самостоятельным преимуществом метрики.'))
    paper.append(ph('3 Экспериментальный протокол'))
    paper.append(pp('Из GMOT-40 [1] по правилу, заданному до инференса, выбраны шесть ID на каждый car-ролик. На ID выделены три референса и два более поздних кандидата. Три реальные сцены дают 18 ID и 90 изображений; игровой car-3 с шестью ID рассматривается отдельно. Эталоны происходят из той же съёмки.'))
    paper.append(pp('Используются три циклических разбиения: одна реальная сцена для обучения, другая для валидации, третья для оценки. Каждый запрос сравнивается с 12 кандидатами сцены: два правильных и десять чужих. Получаются 216 зависимых пар на метод/K. Порог минимизирует FPR на валидации при TPR≥0,70. Метод не выбирается по результатам оцениваемой сцены.'))
    paper.append(PageBreak())
    paper.append(ph('4 Результаты и обсуждение'))
    paper.append(pp('Таблица 1 содержит равновесные средние трёх реальных сцен. TPR — доля принятых правильных пар, FPR — доля принятых чужих, BA=(TPR+1−FPR)/2. V обозначает визуальные признаки, T — текстовые. Полная таблица всех 13 методов, пороги и исходные оценки сохранены в экспериментальном пакете.'))
    selected=['visual_raw_cos','text_raw_cos','fusion_raw_cos','fusion_pca_cos','fusion_full']
    paper.append(table([['Метод','K','TPR, %','FPR, %','BA, %']]+[[label(m),k,pct(stat(m,k)['tpr']),pct(stat(m,k)['fpr']),pct(stat(m,k)['balanced_accuracy'])] for m in selected for k in [1,3]],[200,35,85,85,90]))
    paper.append(Spacer(1,7))
    paper.append(picture(BASE/'balanced_accuracy.png',485,175))
    paper.append(para('Рис. 1. Сравнение модальностей и метрик при одном и трёх референсах. Raw cos — без PCA; остальные метрики используют одинаковые PCA-признаки.','SmallRU'))
    x,y=stat('visual_raw_cos',1),stat('fusion_raw_cos',1)
    a,b=stat('fusion_pca_cos',1),stat('fusion_full',1)
    paper.append(pp(f'При одном референсе добавление текста к raw cosine изменило TPR с {pct(x["tpr"])} до {pct(y["tpr"])}%, FPR с {pct(x["fpr"])} до {pct(y["fpr"])}%. В совместном PCA-пространстве замена косинуса полной метрикой изменила BA с {pct(a["balanced_accuracy"])} до {pct(b["balanced_accuracy"])}%. Эти сравнения не заменяют проверку переносимости рабочего порога.'))
    paper.append(pp('В предшествующей online-абляции на трёх завершённых GMOT-видеопрогонах семантическая проверка не изменила ни одной рамки на 691 кадре, увеличив суммарное время с 53,31 до 198,60 с. На четвёртом ролике оба варианта не инициализировались из-за недостатка SIFT-точек. Поэтому локальный выигрыш признаков нельзя автоматически объявлять улучшением полного поиска.'))
    paper.append(ph('5 Заключение'))
    paper.append(pp('Устойчивый полезный выигрыш текста и полной совместной метрики в данном опыте не установлен. Локальные улучшения сопровождаются переносом порога с потерей полноты или ростом ложных принятий. Вывод ограничен тремя реальными сценами, GT-кандидатами и эталонами из той же съёмки. Нужен подтверждающий опыт на новых ID с отсутствующей целью и независимой съёмкой эталона.'))
    paper.append(ph('Литература'))
    for text in ['[1] Bai H. et al. GMOT-40: A Benchmark for Generic Multiple Object Tracking // CVPR. 2021. P. 6719–6728.',
                 '[2] Oquab M. et al. DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193. 2023.',
                 '[3] Reimers N., Gurevych I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks // EMNLP-IJCNLP. 2019. P. 3982–3992.',
                 '[4] Ledoit O., Wolf M. A well-conditioned estimator for large-dimensional covariance matrices // J. Multivariate Analysis. 2004. Vol. 88(2). P. 365–411.']:
        paper.append(para(text,'SmallRU'))
    paper_path=OUT/'itnt2027_thesis.pdf'
    SimpleDocTemplate(str(paper_path),pagesize=A4,rightMargin=17*mm,leftMargin=17*mm,topMargin=14*mm,bottomMargin=14*mm,title='Визуальные и текстовые признаки в поиске экземпляра объекта',author='В. О. Шабашев').build(paper)
    counts={p.name:len(PdfReader(p).pages) for p in [report_path,paper_path]}
    (OUT/'page_counts.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    print(counts)
    if counts[paper_path.name]!=2:
        raise ValueError('Manuscript must have exactly two pages; adjust layout and render again')


if __name__=='__main__': main()
