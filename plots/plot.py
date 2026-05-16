import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi

metrics_cols = ['faithfulness', 'answer_relevancy', 'context_precision']

plot_df = df[metrics_cols].fillna(0)

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100

plt.figure(figsize=(10, 6))
means = plot_df.mean()
ax = means.plot(kind='bar', color=['#4285F4', '#EA4335', '#FBBC05', '#34A853'][:len(metrics_cols)])

plt.title('Średnie wyniki metryk RAG', fontsize=16, pad=20)
plt.ylabel('Wynik (0.0 - 1.0)', fontsize=12)
plt.xticks(rotation=0)
plt.ylim(0, 1.1) 


for i, v in enumerate(means):
    ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('rag_means.png')
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(plot_df, annot=True, cmap='RdYlGn', vmin=0, vmax=1, center=0.5,
            linewidths=.5, cbar_kws={'label': 'Skala oceny'})

plt.title('Szczegółowa analiza pytań (Heatmap)', fontsize=16, pad=20)
plt.xlabel('Metryki ewaluacyjne', fontsize=12)
plt.ylabel('Indeks pytania w zbiorze', fontsize=12)
plt.tight_layout()
plt.savefig('rag_heatmap.png')
plt.show()

labels = [c.replace('_', ' ').title() for c in metrics_cols]
stats = means.tolist()

angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
stats += stats[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
plt.xticks(angles[:-1], labels, color='grey', size=12)

ax.set_rlabel_position(30)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
plt.ylim(0, 1)

ax.plot(angles, stats, linewidth=2, linestyle='solid', color='#1AA7EC')
ax.fill(angles, stats, color='#1AA7EC', alpha=0.4)

plt.title('Profil wydajności systemu RAG', size=16, y=1.1)
plt.tight_layout()
plt.savefig('rag_radar.png')
plt.show()

print("✅ Wszystkie wykresy zostały wygenerowane i zapisane jako pliki .png!")