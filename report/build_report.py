"""Builds report/report.pdf. Run from the repo root: python report/build_report.py
All numbers are copied from results/ and the analysis scripts' printed output (see results.md in the repo for the log)."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

F = 'C:/Windows/Fonts/'
pdfmetrics.registerFont(TTFont('A', F + 'arial.ttf')); pdfmetrics.registerFont(TTFont('A-B', F + 'arialbd.ttf'))
pdfmetrics.registerFont(TTFont('A-I', F + 'ariali.ttf')); pdfmetrics.registerFont(TTFont('A-BI', F + 'arialbi.ttf'))
pdfmetrics.registerFontFamily('A', normal='A', bold='A-B', italic='A-I', boldItalic='A-BI')

body = ParagraphStyle('b', fontName='A', fontSize=9.6, leading=13.2, alignment=TA_JUSTIFY, spaceAfter=6)
small = ParagraphStyle('s', parent=body, fontSize=8.2, leading=10.6, alignment=0, spaceAfter=3)
cap = ParagraphStyle('c', parent=body, fontSize=8.3, leading=10.8, textColor=colors.HexColor('#333333'), spaceAfter=10, alignment=0)
h1 = ParagraphStyle('h1', fontName='A-B', fontSize=12.5, leading=15, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor('#1a1a1a'))
h2 = ParagraphStyle('h2', fontName='A-B', fontSize=10.2, leading=13, spaceBefore=6, spaceAfter=3)
title = ParagraphStyle('t', fontName='A-B', fontSize=17, leading=21, spaceAfter=4)
sub = ParagraphStyle('sub', fontName='A', fontSize=9.6, leading=13, textColor=colors.HexColor('#444444'), spaceAfter=2)
cellS = ParagraphStyle('cell', fontName='A', fontSize=8.1, leading=10)
cellB = ParagraphStyle('cellb', parent=cellS, fontName='A-B')

def P(t, s=body): return Paragraph(t, s)
def img(path, w=6.5):
    iw, ih = ImageReader(path).getSize(); return Image(path, width=w * inch, height=w * inch * ih / iw)
def fig(path, caption, w=6.5): return KeepTogether([img(path, w), P(caption, cap)])
def table(rows, widths, head=1, align_first_left=True):
    data = [[Paragraph(str(c), cellB if i < head else cellS) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=head)
    t.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#bbbbbb')), ('BACKGROUND', (0, 0), (-1, head - 1), colors.HexColor('#eeeeee')),
                           ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 2.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5)]))
    return t

S = []
S += [P('Hidden decision states and individuality in mouse perceptual decision-making', title),
      P('A replication and extension of GLM-HMM analyses on public IBL behavioral data', sub),
      P('Anas Nadeem &nbsp;·&nbsp; October 2026 &nbsp;·&nbsp; Independent analysis of public data (not a lab project; not peer reviewed)', sub), Spacer(1, 8)]

S += [P('Summary', h1),
      P('I fitted hidden-state models (GLM-HMMs) to 1,154,948 trials from 33 mice in the public International Brain Laboratory (IBL) visual decision task. '
        'The goal was to reproduce the main finding of Ashwood et al. (2022), that mice switch between discrete decision strategies, and then to ask two further questions: '
        'whether the inferred states belong to individual mice, and whether they are reproducible. The main results are:'),
      P('1. <b>Hidden states help in every mouse.</b> On held-out sessions, a 3-state model beat a single-state model in 33 of 33 mice (+0.113 bits/trial on all trials; +0.095 on a replication restricted to the 50/50 trials Ashwood et al. used, against their reported +0.13).'),
      P('2. <b>The states resemble the published ones</b> (a high-stimulus-weight "engaged" state plus side-biased states), but the match is quantitatively weaker than the original paper. In addition, when all trials are used, the biased states are strongly tied to the block structure of the task (section 5), so they cannot be read as disengagement alone.'),
      P('3. <b>Each mouse is best predicted by its own model.</b> This held in 33 of 33 mice with a random split of sessions and in 31 of 33 when models were trained on early sessions and tested on later ones, and it was not a lab effect. The same was true for a plain GLM without hidden states, so individuality does not require hidden states. It was sharper in the random split than in the early/late split.'),
      P('4. <b>States are partly reproducible.</b> Random restarts that reach the same likelihood recover the same states; fits to two halves of one mouse\'s sessions resemble each other more than fits to different mice, but the match is noisy (the nearest match was the same mouse in 39 to 48% of mice for K=3 with interleaved sessions, against 3% by chance).'),
      P('<b>What this is not.</b> It is not a new discovery. Ashwood et al. already reported mouse-specific fits, and Mohammadi, Ashwood and Pillow (2025) already analyze these data with a more sophisticated dynamic model. I did not complete the planned analysis of drift across days. See sections 8 and 9.')]

S += [P('1. Background', h1),
      P('In the IBL task, a head-fixed mouse sees a grating on the left or right at one of several contrasts and reports its side by turning a wheel (IBL et al., 2021). '
        'After training, sessions begin with 90 trials in which the stimulus is equally likely on either side, followed by blocks in which it appears on one side 80% or 20% of the time. '
        'Classical analyses summarize behavior with a single psychometric curve plus a "lapse rate". Ashwood et al. (2022) argued instead that choices come from a small number of hidden states, '
        'each with its own logistic regression (a GLM), and that the state persists for tens of trials. Their model is a GLM-HMM: a hidden Markov model whose emissions are Bernoulli GLMs. '
        'For 37 IBL mice, using only the first 90 trials of each session, they reported that a 3-state model improved held-out log-likelihood by 0.13 bits/trial over one state, '
        'that the median mouse spent 69% of its trials in an engaged state (median dwell 24 trials, versus 13 and 12 for the two biased states), and that the states were consistent across mice. '
        'Mohammadi, Ashwood and Pillow (2025) extended the model so that weights and transitions can change across sessions, and reported four states (engaged-left, engaged-right, biased-left, biased-right) in IBL data with biased blocks.')]

S += [P('2. Data and cohort', h1),
      P('Behavioral data came from the public IBL server through the ONE API. I selected mice with at least 30 biased/ephys-protocol sessions and capped each lab at 5 mice (longest histories first), so that no single lab dominates. '
        'The cohort has 33 mice from 9 labs and 1,552 sessions (1,154,948 trials; median 743 trials per session). Trials without a response (0.5%) were dropped, leaving 1,149,100. '
        '193 early sessions (mainly one lab) were stored in an older per-file format and were recovered separately so that no mouse lost part of its history. '
        'The sign convention for choices was checked from the data (choice −1 is the rewarded response to a right-side stimulus). Median accuracy at non-zero contrast was 0.85 (range 0.68 to 0.93).'),
      fig('figures/psychometric_cohort.png', 'Figure 1. Psychometric curves. Gray: individual mice; black: average. Right: pooled curves by block type. Mice shift their choices toward the more likely side, so they use the block prior.', 6.5)]

S += [P('3. Methods', h1),
      P('<b>Baseline GLM.</b> One logistic regression per mouse predicting a right choice from signed contrast, previous choice (±1), win-stay-lose-switch (previous choice × previous outcome) and a bias term. '
        '<b>GLM-HMM.</b> K hidden states with a Markov transition matrix; in state k, P(right) = sigmoid(w<sub>k</sub> · x). Sessions are independent sequences. '
        'The fitting code (EM with scaled forward-backward in numba; the M-step is a weighted logistic regression per state with mild L2 regularization) was implemented for this project, not taken from an existing package. '
        'A synthetic test with a known two-state model recovered the weights (6.18, 0.5, 2.0 against true 6, 0.5, 2.0), the transition probabilities (0.956 and 0.97 against 0.97 and 0.96) and the state on 92.8% of trials, and the likelihood never decreased across EM iterations. '
        '<b>Evaluation.</b> All model comparisons use 5-fold cross-validation grouped by session, reporting held-out log-likelihood in bits per trial (0 is perfect; −1 is a coin flip). '
        'For each fit, the best of 3 to 5 random restarts by training likelihood was kept. States are compared across fits by one-to-one matching on weights (Hungarian algorithm). '
        'Main analyses use K=3; K=4 is a robustness check. Code: github repository, scripts 01 to 23.')]

S += [P('4. Results: hidden states and the number of states', h1),
      P('A plain GLM reached −0.714 bits/trial with the stimulus alone and −0.640 after adding previous choice; win-stay-lose-switch added almost nothing (−0.638). '
        'Adding hidden states improved held-out likelihood in every mouse at every step from K=1 to K=4 (33 of 33 mice each time), and K=5 and K=6 still improved most mice. The gains shrink but do not stop:'),
      table([['States K', '1', '2', '3', '4', '5', '6'],
             ['Mean held-out score (bits/trial)', '−0.639', '−0.549', '−0.526', '−0.515', '−0.511', '−0.507'],
             ['Gain over previous K', '', '+0.090', '+0.023', '+0.011', '+0.004', '+0.003']], [2.4, .7, .7, .7, .7, .7, .7]),
      Spacer(1, 4),
      P('Because there is no clear plateau, I chose K=3 as the primary model for interpretability and diminishing returns (as Ashwood et al. did), and not because the held-out score stopped improving. '
        'Part of the continued gain may reflect slow changes across days that a single set of fixed states cannot follow.'),
      P('For K=3 (medians over mice, states ranked by stimulus weight), the engaged-like state had a stimulus weight of 21.0, 58% occupancy and a typical stay of 27 trials. '
        'The other two had stimulus weights of 3.5 and 1.9, occupancies of 22% and 16%, and stays of 20 and 10 trials, with large left or right bias in most mice. '
        'The plain GLM\'s average stimulus weight (5.9) mixes these states, which is why one set of weights fits poorly.'),
      fig('figures/states_K3_K4.png', 'Figure 2. Inferred states in all 33 mice. Each dot is a state (size = occupancy; color = rank by stimulus weight). One or two states follow the stimulus strongly; the rest ignore it and lean left or right.', 6.4),
      fig('figures/state_timelines.png', 'Figure 3. Two example mice (K=3). Top: posterior probability of each state across one session. Bottom: share of each state session by session.', 6.4)]

S += [P('5. A confound: block structure and the biased states', h1),
      P('My main fits use all trials but do not give the model the block (the 20/80 prior) as an input. Mice do track the block (Figure 1), so a "biased" state could simply be the mouse following the prior. '
        'I tested this directly. For low-stimulus, side-biased states, occupancy was higher in the block favoring the state\'s side in 46 of 47 states at K=3 (mean +0.15, on a mean overall occupancy of 0.21) and in 54 of 58 at K=4. '
        'So part of what the model calls a biased state reflects block-following, not only disengagement. This likely accounts for part of the difference from Ashwood et al., and it is consistent with Mohammadi et al. finding left- and right-biased versions of engaged states in block data.'),
      fig('figures/block_control.png', 'Figure 4. Occupancy of side-biased states when the block favors the state\'s side (y) versus the other side (x). Points above the diagonal show the state follows the block.', 5.4)]

S += [P('6. Closer replication of Ashwood et al.', h1),
      P('To compare more directly with the original paper, I refitted using only the first 90 trials of each session (the 50/50 block; 138,559 trials, median 4,137 per mouse) and their input features (stimulus, bias, previous choice, previous stimulus side). '
        'The 3-state model beat the 1-state model in 33 of 33 mice (+0.095 bits/trial; Ashwood et al.: +0.13). The gains per added state were 0.075, 0.020 and 0.008, so the curve flattens near three states as they reported, although 26 of 33 mice still preferred K=4 by a small margin. '
        'In the full-data K=3 fits, the median engaged occupancy was 59% (published: 69%), the median engaged dwell time was 22.8 trials (published: 24), and the other two states had medians of 17.8 and 6.9 trials (published: 13 and 12). '
        'In 24 of 33 mice the two non-engaged states had biases of opposite sign, as expected for biased-left and biased-right states.'),
      fig('figures/replication_vs_all_trials.png', 'Figure 5. Held-out gain over a single state as a function of K, for the Ashwood-style 50/50 subset and for all trials. The star marks the published K=3 value.', 4.2)]

S += [P('7. Results: individuality and reproducibility', h1),
      P('<b>Individuality.</b> For every mouse and fold, I fitted a model on that mouse\'s training sessions and scored every mouse\'s model on every mouse\'s held-out sessions. '
        'For each test mouse, I compared its own model with the average of the 32 other models on the same data, which controls for some mice being easier to predict. '
        'With a random split of sessions, the own model won in 33 of 33 mice (mean advantage +0.056 bits/trial at K=3) and ranked first among 33 models in 30 mice. '
        'Models from lab-mates were no better than models from other labs (difference −0.0015 bits/trial), so this is not a rig or lab effect. '
        'The plain 1-state GLM showed it as well (33 of 33 mice, +0.072), so individuality is present without hidden states; the hidden-state models made the own model the top-ranked more consistently (30 vs 21 mice).'),
      table([['Split', 'K', 'Own beats others', 'Mean advantage (bits)', 'Mean own rank (chance 17)', 'Rank 1', 'Top 5'],
             ['Random sessions', '1', '33/33', '+0.072', '1.9', '21', '32'], ['Random sessions', '3', '33/33', '+0.056', '1.6', '30', '32'], ['Random sessions', '4', '33/33', '+0.055', '1.6', '28', '31'],
             ['Early 60% → late 40%', '1', '31/33', '+0.084', '7.6', '3', '18'], ['Early 60% → late 40%', '3', '31/33', '+0.061', '5.3', '9', '24'], ['Early 60% → late 40%', '4', '31/33', '+0.057', '5.2', '10', '23']],
            [1.45, .35, 1.0, 1.15, 1.35, .5, .5]),
      Spacer(1, 4),
      P('When models were trained on early sessions and tested on later ones, the average advantage was similar, but the own model was the single best in only 9 mice (K=3). '
        'So individuality persists over time but is less sharp. Random splits interleave test sessions with training sessions, which can inflate the advantage, so the early/late result is the more conservative one. '
        'The own model\'s score fell from −0.510 on its training period to −0.556 on later sessions; this mixes ordinary overfitting with change over time and I could not separate them.'),
      fig('figures/individuality_summary.png', 'Figure 6. Individuality at K=3. Left and middle: own-model score versus the mean of other mice\'s models (points above the diagonal favor the own model). Right: rank of the own model.', 6.5),
      P('<b>Reproducibility of the states.</b> Across 5 random restarts, those within 0.001 bits/trial of the best likelihood recovered the same states (median unstandardized weight distance 0.008 at K=3), whereas restarts that settled at worse likelihoods found different states (median distance about 4.3, close to the typical distance between different mice, 5.7). '
        'K=3 was steadier than K=4 (46% vs 26% of restarts matching the best fit closely). '
        'For split-half reproducibility, I fitted each mouse separately to two halves of its sessions and matched states on standardized weights. With interleaved sessions at K=3, a mouse\'s two halves were closer to each other than to other mice in 31 of 33 mice (median distance 0.66 vs 1.54), '
        'and the nearest fit among all 33 was the same mouse in 39% and 48% of cases (two directions; chance 3%). With early versus late halves this dropped to 24% and 15% (K=3). '
        'So the states are a partly stable property of the animal, but not a precise fingerprint, and they appear to change across the training history. Each half has half the data, so estimation noise accounts for some of the mismatch.')]

S += [P('8. How the findings compare with prior work', h1),
      table([['Topic', 'Published', 'This analysis', 'Agreement'],
             ['Hidden states beat one state', 'Yes in all 37 IBL mice; +0.13 bits/trial at K=3', 'Yes in 33/33 mice; +0.095 (50/50 trials), +0.113 (all trials)', 'Agrees; gain is smaller'],
             ['Number of states', 'Flattens near 3; K=3 chosen for simplicity', '50/50 data: flattens near 3 (K=4 still better in 26/33). All trials: no plateau through K=6', 'Partly agrees'],
             ['State types', 'One engaged, two biased (left, right)', 'Same pattern: 24/33 mice with opposite-sign biased states (50/50 data)', 'Agrees'],
             ['Engaged occupancy / dwell', 'Median 69%; 24 vs 13 and 12 trials', 'Median 59%; 22.8 vs 17.8 and 6.9 trials', 'Same direction; weaker'],
             ['Four states in block data', 'Mohammadi et al. 2025: engaged-L/R and biased-L/R', 'K=4 best in all-trial CV; biased states track the block', 'Consistent'],
             ['Consistency across mice', '"Remarkably consistent" state types; fits per mouse', 'Same state types but mouse-specific weights: own model beats others', 'Complementary'],
             ['Lab differences', 'IBL 2021: none after training', 'Lab-mates\' models no better than others\'', 'Agrees'],
             ['Drift across days', 'Dynamic model of Mohammadi et al. 2025', 'Only indirect hints (early/late results); not analyzed', 'Not tested']],
            [1.35, 1.95, 2.25, 1.0]),
      Spacer(1, 4),
      P('Overall, the findings agree qualitatively with Ashwood et al. and are somewhat weaker quantitatively. I did not control for the remaining differences: my cohort differs from theirs, my fits start from random initializations rather than a pooled model, and my feature set differs slightly (win-stay-lose-switch in the main fits). '
        'The individuality test is complementary to their finding of consistent state types rather than in conflict with it: it asks whether one mouse\'s fitted model predicts that mouse better than other mice\'s models do.')]

S += [P('9. Limitations', h1),
      P('(1) This is an independent re-analysis of public data with one cohort of 33 mice from 9 labs and one model family; it has not been peer reviewed and I make no claim of novelty. '
        '(2) The main fits include the 20/80 blocks without block information, which affects how the biased states should be interpreted (section 5). '
        '(3) K was chosen for interpretability; held-out likelihood kept improving. '
        '(4) Individuality was tested with two split schemes; I did not decompose which parameters (bias, stimulus sensitivity, history, transitions) carry it. '
        '(5) Weight-distance comparisons depend on the standardization and on the matching procedure. '
        '(6) Estimation noise and real change over time are confounded in the early/late comparisons. '
        '(7) <b>The planned analysis of drift across days was not completed</b>: I only observed that engaged-state share fell late in the history of two example mice and that early/late results were weaker than interleaved ones. These are observations, not tests, and later sessions may differ in protocol or conditions. '
        '(8) The optional comparison with neural data was not attempted. '
        '(9) The fitting code was validated on synthetic data with a known answer, results were compared with the published papers, and all numbers in this report come from the scripts in the repository.')]

S += [P('References', h1),
      P('Ashwood, Z. C., Roy, N. A., Stone, I. R., The International Brain Laboratory, Urai, A. E., Churchland, A. K., Pouget, A., &amp; Pillow, J. W. (2022). Mice alternate between discrete strategies during perceptual decision-making. <i>Nature Neuroscience</i>, 25, 201-212.', small),
      P('Mohammadi, Z., Ashwood, Z. C., &amp; Pillow, J. W. (2025). Identifying the factors governing internal state switches during nonstationary sensory decision-making. <i>Nature Communications</i>. (First posted on bioRxiv, 2024.)', small),
      P('The International Brain Laboratory (2021). Standardized and reproducible measurement of decision-making in mice. <i>eLife</i>, 10, e63711.', small),
      P('Behavioral data: IBL public database (openalyx.internationalbrainlab.org), accessed through the ONE API. Code and results: see the repository README.', small),
      P('Details of the cited papers were taken from their open-access full texts; I did not re-derive their numbers.', small)]

def footer(c, d):
    c.saveState(); c.setFont('A', 7.5); c.setFillColor(colors.HexColor('#777777'))
    c.drawString(0.8 * inch, 0.5 * inch, 'GLM-HMM analysis of IBL mouse behavior - independent analysis, not peer reviewed')
    c.drawRightString(7.7 * inch, 0.5 * inch, f'Page {d.page}'); c.restoreState()

doc = SimpleDocTemplate('report/report.pdf', pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch, topMargin=0.75 * inch, bottomMargin=0.8 * inch,
                        title='Hidden decision states and individuality in mouse perceptual decision-making', author='Anas Nadeem')
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print('built report/report.pdf')
