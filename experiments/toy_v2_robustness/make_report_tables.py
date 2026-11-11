"""Format the two frozen result versions; never fit or recompute statistics."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'experiments/results/toy_v2_robustness'
TABLES = ROOT / 'paper/tables_toy_v2_robustness'


def interval(value, limits, tex=False):
    if tex:
        return rf'\shortstack{{{value:.4f}\\({limits[0]:.4f}, {limits[1]:.4f})}}'
    return f'{value:.4f} [{limits[0]:.4f}, {limits[1]:.4f}]'


def metric(m, key, tex=False):
    return interval(m[key], m[key + '_ci95'], tex)


def tex_table(headers, rows, columns):
    return ('% Generated only from the committed numerical summaries.\n'
            '\\begin{tabular}{' + columns + '}\n\\toprule\n' +
            ' & '.join(headers) + r' \\' + '\n\\midrule\n' +
            '\n'.join(' & '.join(row) + r' \\[3pt]' for row in rows) +
            '\n\\bottomrule\n\\end{tabular}\n')


def markdown_table(headers, rows):
    return ('| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join('---' for _ in headers) + ' |\n' +
            '\n'.join('| ' + ' | '.join(row) + ' |' for row in rows) + '\n')


def main():
    old = json.loads((RESULTS / 'summary.json').read_text())
    new = json.loads((RESULTS / 'amendment_1/summary.json').read_text())
    TABLES.mkdir(parents=True, exist_ok=True)
    primary = new['metrics']['common']['0.05']
    md = ['# Toy V2: both frozen analysis versions', '',
          'Evaluation seeds 100–199, sigma=0.05, common projection. '
          'All brackets are paired-seed percentile 95% bootstrap intervals (1000 draws).', '',
          '## Amendment 1: main analysis', '']
    auc_headers = ['Control', 'H1 AUC', 'G AUC', 'G+H1 AUC', 'Delta AUC']
    accuracy_headers = ['Control', 'H1 accuracy', 'G accuracy', 'G+H1 accuracy']
    auc_tex, auc_md, accuracy_tex, accuracy_md = [], [], [], []
    for control in ('GAUSS', 'UNIFORM', 'LINE'):
        m = primary[control]
        for rows, is_tex in ((auc_tex, True), (auc_md, False)):
            rows.append([control] + [metric(m[arm], 'auc', is_tex) for arm in ('H1', 'G', 'G+H1')] +
                        [interval(m['delta_auc']['estimate'], m['delta_auc']['ci95'], is_tex)])
        for rows, is_tex in ((accuracy_tex, True), (accuracy_md, False)):
            rows.append([control] + [metric(m[arm], 'accuracy', is_tex) for arm in ('H1', 'G', 'G+H1')])
    (TABLES / 'amended_auc.tex').write_text(tex_table(
        ['Control', '$H_1$', '$G$', '$G+H_1$', '$\\Delta$AUC'], auc_tex, 'lcccc'))
    (TABLES / 'amended_accuracy.tex').write_text(tex_table(
        ['Control', '$H_1$', '$G$', '$G+H_1$'], accuracy_tex, 'lccc'))
    md += [markdown_table(auc_headers, auc_md), markdown_table(accuracy_headers, accuracy_md),
           'Categories: ' + '; '.join(f'{key}: **{value}**' for key,value in new['categories'].items()), '']
    names = {'H1': '$H_1$', 'g1': '$g_1$', 'g2': '$g_2$', 'g3': '$g_3$',
             'g1_second_half': '$g_{1,\\mathrm{half}}$'}
    for version, conditions, filename, title in (
        ('amended', primary, 'amended_geometry.tex', 'Amended individual statistics'),
        ('original', old['metrics']['common']['0.05'], 'original.tex', 'Original committed analysis')):
        tex_rows, md_rows = [], []
        for comparison, stats in conditions.items():
            for stat in names:
                m = stats[stat]
                tex_rows.append([comparison, names[stat], metric(m, 'auc', True), metric(m, 'accuracy', True)])
                md_rows.append([comparison, stat, metric(m, 'auc'), metric(m, 'accuracy')])
        (TABLES / filename).write_text(tex_table(
            ['Comparison / control', 'Statistic', 'AUC (95\\% CI)', 'Accuracy (95\\% CI)'], tex_rows, 'llcc'))
        md += ['## ' + title, '', markdown_table(['Comparison / control', 'Statistic', 'AUC', 'Accuracy'], md_rows)]
    md += ['Original R1: **' + old['categories']['R1'] + '**. Original R2: **' + old['categories']['R2'] + '**.', '',
           'The amended main analysis and original analysis use different pre-specified decision rules. '
           'Neither implies an LLM, user-identity, or temporal-return result.', '']
    (RESULTS / 'amendment_1/main_tables.md').write_text('\n'.join(md))


if __name__ == '__main__':
    main()
