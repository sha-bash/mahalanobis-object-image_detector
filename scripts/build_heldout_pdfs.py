"""Append held-out evidence to the report; write a revised two-page manuscript."""
import json
from pathlib import Path
from types import SimpleNamespace

from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, PageBreak, Spacer

from build_research_pdfs import para, table, picture, footer

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'experiments/evidence/heldout_v1'
OUT = ROOT/'output/research_20261010_heldout'
LABELS = {'geometry':'Без текста', 'isotropic':'Изотропный Махаланобис',
          'text_cosine':'Косинус, PCA', 'text_full':'Полный Махаланобис, PCA'}


def number(value, percent=False):
    if value is None:
        return '—'
    return f'{value*(100 if percent else 1):.1f}' if percent else f'{value:.3f}'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads((BASE/'summary.json').read_text())
    checks = json.loads((BASE/'semantic_checks.json').read_text())
    rows, macro = data['rows'], data['macro']
    primary = data['comparisons']['primary']
    delta = primary['recall']
    old = ROOT/'output/research_20261010/research_report.pdf'
    stats = {r['variant']:r for r in macro}
    g, m = stats['geometry'], stats['text_full']
    overview = (f'На восьми новых видео средняя полнота без текста равна {number(g["recall"], True)}%, '
                f'с полной обученной метрикой — {number(m["recall"], True)}%. '
                f'Изменение составляет {number(delta["mean"], True)} процентного пункта. '
                f'Средний IoU: {number(g["mean_iou_present"])} → {number(m["mean_iou_present"])}. '
                f'Улучшений / ухудшений / равенств по полноте: {delta["positive"]} / {delta["negative"]} / {delta["ties"]}. '
                f'Одностороннее p={delta["one_sided_p"]:.4f}.')
    decision = ('H0 не отвергается при α=0,05. Это не доказательство равенства методов и не вероятность истинности H0.'
                if delta['one_sided_p'] >= .05 else
                'Получено основание отвергнуть H0 при α=0,05 только при выполнении предположения о независимости видео. Практическая полезность требует проверки ложных принятий.')
    bootstrap_warning = (' Все наблюдавшиеся разности одинаковы: вырожденный bootstrap-интервал отражает только эту выборку, а не отсутствие неопределённости на новых видео.'
                         if delta['bootstrap_degenerate'] else '')
    summary_table = [['Метод', 'Полнота, %', 'Средний IoU', 'TP', 'FP', 'Чужой ID']]
    summary_table += [[LABELS[r['variant']], number(r['recall'], True), number(r['mean_iou_present']), r['tp'], r['fp'], r['wrong_instance']] for r in macro]
    report = [para('Дополнение: проверка на новых видео', 'TitleRU'),
              para('10 октября 2026 · протокол e36a8f8 · результаты предыдущего опыта сохранены'),
              para('Предыдущие 12 страниц описывают первый этап. Это дополнение оценивает уже полный поиск и сопровождение на восьми ранее не использованных клипах GMOT-40: четыре авиационных и четыре с судами. Эталон — одна вырезка нулевого кадра; алгоритм не получает рамку в оцениваемом видео.'),
              para(overview), para(decision), table(summary_table, [180, 65, 65, 45, 45, 65]),
              para(f'Отдельный компромисс: исходный изотропный фильтр снизил FP с {g["fp"]} до {stats["isotropic"]["fp"]}, но TP с {g["tp"]} до {stats["isotropic"]["tp"]}. Суммарная по кадрам precision выросла с {number(g["micro_precision"], True)}% до {number(stats["isotropic"]["micro_precision"], True)}%. Здесь FP — рамки с недостаточным IoU на присутствии цели, а не ложные срабатывания при её отсутствии.'),
              para('Полнота и IoU в таблице усреднены равновесно по видео. TP/FP и чужие ID суммируются по кадрам. Число кадров не является числом независимых наблюдений. Отказ на кадре присутствия даёт IoU=0 и FN.'),
              para('Что означает независимая проверка', 'HeadRU'),
              para('Новые клипы не участвовали в обучении проекции, ковариации или выборе порога. Все параметры перенесены с car-0/car-1. Однако это тот же публичный набор; независимость событий между новыми клипами, идентичность физических объектов и отсутствие пересечения с предобучением моделей не гарантированы. Это отложенная проверка переноса на другие виды техники, а не подтверждение на новых автомобилях.'),
              para('Отсутствие цели не оценивалось: отсутствие ID в разметке не принималось за доказательство отсутствия объекта. Неразмеченные по цели кадры исключены из расчёта качества, но обработаны алгоритмом.'), PageBreak()]
    report += [para('Гипотеза, метрики и фиксированные условия', 'TitleRU'),
               para('Основная нулевая гипотеза: медиана изменения полноты при добавлении текстовой проверки с полной обученной метрикой Махаланобиса не больше нуля. Альтернатива — положительная медиана. Основное сравнение задано до новых запусков: полный Махаланобис против геометрии.'),
               para('ΔR<sub>j</sub> = R<sub>М,j</sub> − R<sub>Г,j</sub>; H<sub>0</sub>: median(ΔR) ≤ 0; H<sub>1</sub>: median(ΔR) &gt; 0.'),
               para('IoU(B,G) = |B ∩ G| / (|B| + |G| − |B ∩ G|). R = Σ 1[IoU ≥ 0,5] / N<sub>present</sub>. Средний IoU = Σ IoU / N<sub>present</sub>; для отсутствующей рамки IoU=0. Precision = TP / (TP+FP). Неверная рамка на присутствии одновременно даёт FP и FN.'),
               para('Косинус: d<sub>C</sub>(a,b)=1−aᵀb. Махаланобис: d<sub>M</sub>(a,b)=√[(a−b)ᵀΣ⁻¹(a−b)]. Оба варианта получают одинаковые нормированные PCA8-проекции текстов. Матрица обучена по разностям внутри ID на car-0, регуляризация Ledoit–Wolf плюс 10⁻⁴I; пороги взяты с car-1, K=1.'),
               para('Исходный изотропный вариант использует Σ=0,01I и порог 6, эквивалентный косинусному расстоянию ≤0,18 для нормированных эмбеддингов. Его нельзя представлять как самостоятельный выигрыш метрики.'),
               para('SIFT/RANSAC и optical flow одинаковы; ширина ≤1920; seed2027; проверки при захвате и каждые 10 секунд рабочего времени. Взяты первые 300 исходных кадров; нулевой используется только для эталона. Цель выбрана по минимальному ID с рамкой ≥40×20 и ≥30 размеченными кадрами, без просмотра качества метода.'),
               para('Статистическая процедура', 'HeadRU'),
               para('Односторонний консервативный знаковый тест: число положительных ΔR среди всех восьми клипов сравнивается с Binomial(8,0,5). Равенство считается отсутствием улучшения. Исходный план предполагал исключение равенств; это исправлено до агрегирования, после просмотра первых двух клипов, поскольку полнота дискретна. Поправка документирована отдельно, исходный протокол сохранён.'),
               para(f'Описательный парный bootstrap по целым видео, 10 000 повторов: 95%-интервал средней ΔR [{number(delta["descriptive_bootstrap_95"][0], True)}; {number(delta["descriptive_bootstrap_95"][1], True)}] п.п. Это интервал наблюдавшихся клипов при сильных предположениях, не доказательство эквивалентности. Восемь видео не дают надёжной оценки редких ошибок.'+bootstrap_warning), PageBreak()]
    for prefix, title in [('airplane', 'Авиационные последовательности'), ('boat', 'Последовательности с судами')]:
        report += [para(title, 'TitleRU')]
        grid = [['Видео / метод', 'N', 'TP', 'FP', 'Полнота, %', 'IoU', 'Чужой ID']]
        for r in rows:
            if r['sequence'].startswith(prefix):
                grid.append([r['sequence']+' / '+LABELS[r['variant']], r['present'], r['tp'], r['fp'], number(r['recall'], True), number(r['mean_iou_present']), r['wrong_instance']])
        report += [table(grid, [210, 40, 40, 40, 65, 50, 45]), Spacer(1, 10),
                   para('N — число кадров присутствия. Отказ из-за недостатка геометрических признаков, если он возник, учитывается как нулевые предсказания, а не исключение неудобного видео. Рамка чужого ID учитывается при IoU≥0,5 к размеченному отвлекающему объекту; неразмеченные отвлекающие объекты этим счётчиком не покрываются.'),
                   para('Точные рамки, значения IoU, переходы потери и восстановления и исходные семантические решения доступны в experiments/evidence/heldout_v1. Данные таблицы автоматически читаются из summary.json.')]
        if prefix == 'boat':
            case = {r['variant']:r for r in rows if r['sequence'] == 'boat-2'}
            initial = next(c for c in checks if c['sequence'] == 'boat-2' and c['variant'] == 'text_full' and c['frame'] == 0)
            report += [para('Разбор ошибки на boat-2', 'HeadRU'),
                       para(f'Первый кандидат правильный по геометрии: IoU={initial["target_iou"]:.3f}. Полный Махаланобис дал расстояние {initial["distance"]:.4f} при пороге {initial["threshold"]:.4f} и отклонил его. Полнота: геометрия {number(case["geometry"]["recall"], True)}%, косинус {number(case["text_cosine"]["recall"], True)}%, полный Махаланобис {number(case["text_full"]["recall"], True)}%. Причина пропуска видна в журнале: вместо лодки VLM описала тёмный вытянутый предмет; повторный захват произошёл через 30 кадров. Это наблюдаемый отрицательный пример, а не доказательство универсального преимущества косинуса.')]
        report.append(PageBreak())
    report += [para('Локализация и чувствительность к порогу IoU', 'TitleRU'),
               picture(BASE/'localization_ru.png', width=495, max_height=345),
               para('Рисунок 1. Полнота и средний IoU по видео. Все пропуски учитываются. Совпадающие столбцы означают одинаковый результат, а не пропущенные методы.', 'SmallRU'),
               picture(BASE/'success_iou_ru.png', width=465, max_height=245),
               para('Рисунок 2. Кривая успешности локализации; среднее по восьми видео. Основной порог IoU=0,5 зафиксирован заранее; эта кривая не использовалась для выбора метода или порога.', 'SmallRU'), PageBreak()]
    semgrid = [['Метод', 'Проверок', 'Принято', 'Верных принято', 'Неверных принято', 'Верных отклонено']]
    for variant in LABELS:
        if variant == 'geometry':
            continue
        cs = [c for c in checks if c['variant'] == variant]
        semgrid.append([LABELS[variant], len(cs), sum(c['accepted'] for c in cs),
                        sum(c['accepted'] and c['positive_candidate'] is True for c in cs),
                        sum(c['accepted'] and c['positive_candidate'] is False for c in cs),
                        sum(not c['accepted'] and c['positive_candidate'] is True for c in cs)])
    report += [para('Семантический модуль, затраты и вывод', 'TitleRU'), table(semgrid, [155, 50, 50, 80, 80, 80]),
               para('Верный кандидат здесь означает IoU≥0,5 до решения семантического модуля. Недостаточный IoU может быть ошибкой границ, а не подменой экземпляра. Это условная оценка проверенных кандидатов, не полнота всей системы. После отклонения пути трекеров могут различаться, поэтому наборы проверок не обязательно попарно совпадают.'),
               table([['Метод', 'Время всех запусков, с', 'Новых описаний', 'Описаний из кэша']]+[[LABELS[r['variant']], f'{r["total_wall_sec"]:.2f}', r['caption_cold_descriptions'], r['caption_cache_hits']] for r in macro], [185, 100, 100, 110]),
               para('Описания одинаковых изображений переиспользуются между семантическими вариантами внутри нового опыта. Порядок фиксирован; времена с кэшем нельзя трактовать как доказательство более высокой скорости одной метрики. Память дерева процессов измерялась каждые 0,2 с и сохранена в execution.json.'),
               para(decision), para('Числа подтверждают результат конкретной реализации на зафиксированных данных. Они не обосновывают тезис «Махаланобис всегда лучше» и не опровергают все способы использования текста. Главный следующий опыт — новые автомобили, независимая съёмка эталона, заранее проверенные интервалы отсутствия и достаточно похожих отрицательных экземпляров.'),
               para('Воспроизведение и источники', 'HeadRU'),
               para('Команды: experiments/HELDOUT_REPRODUCE.md. Протокол: HELDOUT_PROTOCOL.md; уточнения: HELDOUT_AMENDMENTS.md. CSV/JSON и параметры запусков: experiments/evidence/heldout_v1. Старые параметры: experiments/evidence/multimodal_v1. Тестовые пороги не оптимизировались.', 'SmallRU'),
               para('GMOT-40: https://spritea.github.io/GMOT40/ и /download.html; Bai et al., CVPR 2021, 6719–6728. Лицензия CC BY-NC-SA 4.0. Авторы набора предоставляют ID и рамки траекторий. Проверка источника: 10.10.2026.', 'SmallRU'),
               para('Биномиальный тест: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html. Реализация в проекте вычисляет точную биномиальную сумму; проверена тестами.', 'SmallRU')]
    appendix = OUT/'heldout_appendix.pdf'
    offset = len(PdfReader(old).pages)
    def continued_footer(canvas, doc):
        footer(canvas, SimpleNamespace(page=doc.page+offset))
    SimpleDocTemplate(str(appendix), pagesize=A4, leftMargin=19*mm, rightMargin=16*mm, topMargin=18*mm, bottomMargin=18*mm).build(report, onFirstPage=continued_footer, onLaterPages=continued_footer)
    writer = PdfWriter()
    writer.append(old)
    writer.append(appendix)
    writer.write(OUT/'research_report.pdf')

    paper = []
    def p(text, style='PaperBody'):
        paper.append(para(text, style))
    p('Отложенная проверка текстовой верификации<br/>при поиске экземпляра объекта на видео', 'PaperTitle')
    p('В. О. Шабашев', 'CenterRU')
    p('Ульяновский государственный технический университет, Ульяновск, Россия', 'CenterRU')
    p(f'<b>Аннотация.</b> Проверено влияние текстовой верификации на поиск заданного экземпляра по одному изображению. На восьми ранее не использованных видео GMOT-40 сравниваются геометрия, VLM/SBERT с косинусом и с расстоянием Махаланобиса. Параметры перенесены с других сцен. Полнота без текста — {number(g["recall"], True)}%, с обученным Махаланобисом — {number(m["recall"], True)}%; средний IoU — {number(g["mean_iou_present"])} и {number(m["mean_iou_present"])}. Обсуждаются переносимость и границы статистического вывода.')
    p('<b>Ключевые слова:</b> поиск экземпляра; текстовая верификация; IoU; Sentence-BERT; расстояние Махаланобиса.')
    p('1 Постановка задачи и гипотеза', 'PaperHead')
    p('На вход подаются одно изображение конкретного объекта и видео. Требуется выдать рамку именно этого экземпляра либо отказ. Геометрия SIFT/RANSAC формирует кандидатов, оптический поток поддерживает сопровождение. Дополнительный модуль независимо описывает эталон и кандидата через VLM, затем сравнивает текстовые признаки. Близость описаний категории не гарантирует идентичности экземпляра.')
    p('Основная H<sub>0</sub>: медиана изменения полноты локализации при добавлении текстовой проверки с полной обученной метрикой не больше нуля. H<sub>1</sub>: изменение положительно. Критерий успеха кадра — IoU≥0,5. Ложное принятие другого экземпляра оценивается отдельно; одного роста полноты недостаточно для вывода о практической полезности.')
    p('2 Метод и контролируемое сравнение', 'PaperHead')
    p('Использованы независимые описания Qwen3.5:0.8b без передачи текста эталона кандидату и нормированные эмбеддинги all-MiniLM-L6-v2 [2]. PCA до восьми компонент обучена на car-0; проекции повторно нормированы. На одинаковых признаках сравниваются косинусное расстояние и полная ковариация разностей внутри ID с регуляризацией Ledoit–Wolf [3].')
    p('d<sub>C</sub>(a,b)=1−aᵀb; &nbsp; d<sub>M</sub>(a,b)=√[(a−b)ᵀΣ⁻¹(a−b)].')
    p('Пороги перенесены с car-1 из предшествующего опыта: минимальная доля ложных принятий при полноте ≥0,70 на валидации. Тест не используется для настройки. Дополнительно сохранён исходный вариант Σ=0,01I, τ=6, эквивалентный d<sub>C</sub>≤0,18 на нормированных исходных эмбеддингах. Он не является самостоятельным преимуществом метрики.')
    p('3 Данные и протокол', 'PaperHead')
    p('Восемь новых клипов GMOT-40 [1]: airplane-0…3 и boat-0…3. Для каждого фиксирован минимальный ID с рамкой ≥40×20 на кадре 0 и не менее 30 размеченными кадрами. Эталон — вырезка нулевого кадра; кадр исключён из оценки. Обрабатываются следующие кадры до исходного 299 включительно либо до конца ролика. Начальная рамка на видео алгоритму не передаётся.')
    p(f'Оценено {g["present"]} кадров присутствия на вариант; {g["ignored"]} кадров без разметки цели обработаны, но не объявлены отрицательными. Эталон происходит из той же съёмки. Это перенос на другие виды техники; проверка на новых автомобилях и независимой съёмке эталона остаётся отдельной задачей.')
    paper.append(PageBreak())
    p('4 Результаты', 'PaperHead')
    paper.append(table([['Метод', 'Полнота, %', 'Средний IoU', 'FP', 'Чужой ID']]+[[LABELS[r['variant']], number(r['recall'], True), number(r['mean_iou_present']), r['fp'], r['wrong_instance']] for r in macro], [210, 75, 75, 50, 65]))
    p('Таблица 1. Полнота и IoU усреднены по видео; ошибки — суммы по кадрам. Пропуски имеют IoU=0. FP — выданная рамка с IoU&lt;0,5 к цели; при присутствии такая ошибка также даёт FN.', 'SmallRU')
    paper.append(picture(BASE/'success_iou_ru.png', width=435, max_height=235))
    p('Рис. 1. Успешность локализации при изменении порога IoU. Подбор рабочего порога по этой кривой не выполнялся.', 'SmallRU')
    p(overview)
    p(f'Исходный изотропный фильтр уменьшил FP с {g["fp"]} до {stats["isotropic"]["fp"]} при потере {g["tp"]-stats["isotropic"]["tp"]} правильных кадров; precision по кадрам {number(g["micro_precision"], True)} → {number(stats["isotropic"]["micro_precision"], True)}%. Это локальный компромисс полноты и точности, а не превосходство метрики над эквивалентным косинусом.')
    p(decision)
    p('5 Ограничения и вывод', 'PaperHead')
    p('Клипы не участвовали в настройке, но относятся к одному набору; независимость событий и физических объектов между ними не гарантирована. Эталон взят из той же сцены. Проверенных интервалов отсутствия цели нет, поэтому частота ложных срабатываний на отсутствии не оценивается. Единица статистического анализа — видео, а не соседний кадр.')
    p('Знаковый тест использует все восемь разностей; равенства считаются отсутствием улучшения. Это уточнение дискретного случая внесено после первых двух клипов, до агрегирования, и сохранено отдельно от исходного протокола. Численный результат относится к данным конфигурациям; утверждать универсальное превосходство семантики или Махаланобиса нельзя.')
    p('Литература', 'PaperHead')
    p('[1] Bai H. et al. GMOT-40: A Benchmark for Generic Multiple Object Tracking // CVPR. 2021. P. 6719–6728.', 'SmallRU')
    p('[2] Reimers N., Gurevych I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks // EMNLP-IJCNLP. 2019. P. 3982–3992.', 'SmallRU')
    p('[3] Ledoit O., Wolf M. A well-conditioned estimator for large-dimensional covariance matrices // Journal of Multivariate Analysis. 2004. Vol. 88(2). P. 365–411.', 'SmallRU')
    SimpleDocTemplate(str(OUT/'itnt2027_thesis.pdf'), pagesize=A4, leftMargin=19*mm, rightMargin=16*mm, topMargin=16*mm, bottomMargin=14*mm).build(paper)
    counts = {f.name:len(PdfReader(f).pages) for f in OUT.glob('*.pdf')}
    (OUT/'page_counts.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(counts)
    assert counts['itnt2027_thesis.pdf'] == 2, counts


if __name__ == '__main__':
    main()
