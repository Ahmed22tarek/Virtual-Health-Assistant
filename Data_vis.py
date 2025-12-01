import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 120
custom_palette = sns.color_palette("mako", 10)

df = pd.read_csv('mtsamples.csv')

df = df.dropna(subset=['transcription', 'medical_specialty'])

df['word_count'] = df['transcription'].apply(lambda x: len(str(x).split()))

# ==========================================
# الشكل الأول: أكثر التخصصات الطبية شيوعاً
# ==========================================
plt.figure(figsize=(12, 8))
top_specialties = df['medical_specialty'].value_counts().nlargest(15).index
ax = sns.countplot(y='medical_specialty',
                   data=df[df['medical_specialty'].isin(top_specialties)],
                   order=top_specialties,
                   palette='viridis')
plt.title('Top 15 Medical Specialties by Number of Records', fontsize=16, fontweight='bold')
plt.xlabel('Number of Transcriptions', fontsize=12)
plt.ylabel('Medical Specialty', fontsize=12)
sns.despine(left=True, bottom=True)
plt.show()

# ==========================================
# الشكل الثاني: توزيع طول التقارير الطبية (Density Plot)
# ده مهم عشان تعرف الـ AI بتاعك هيستقبل نصوص طويلة ولا قصيرة
# ==========================================
plt.figure(figsize=(12, 6))
sns.histplot(df['word_count'], bins=50, kde=True, color='#2c3e50', alpha=0.6)
plt.title('Distribution of Medical Report Word Counts', fontsize=16, fontweight='bold')
plt.xlabel('Word Count per Report', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xlim(0, 1500)
sns.despine()
plt.show()

# ==========================================
# الشكل الثالث: مقارنة طول التقارير بين التخصصات (Box Plot)
# هل تقارير الجراحة أطول من تقارير الأشعة؟
# ==========================================
plt.figure(figsize=(14, 8))
top_10_spec = df['medical_specialty'].value_counts().nlargest(10).index
subset_df = df[df['medical_specialty'].isin(top_10_spec)]
sns.boxplot(x='word_count', y='medical_specialty', data=subset_df, palette="coolwarm")
plt.title('Word Count Distribution Across Top 10 Specialties', fontsize=16, fontweight='bold')
plt.xlabel('Word Count', fontsize=12)
plt.ylabel('Specialty', fontsize=12)
plt.show()

