import pandas as pd
import os


raw = pd.read_excel(r'raw/OCTA-DPN-第一部分.xlsx')
raw = raw.iloc[:, :14]

raw['upper'] = pd.eval('raw.正中神经1 < 50 | raw.正中神经2 < 50')
raw['lower'] = pd.eval('raw.胫神经1 < 40 | raw.胫神经2 < 40')
raw['symptom'] = pd.eval('raw.症状')
raw['sign'] = pd.eval('raw.踝反射 | raw.振动觉 | raw.压力觉 | raw.温度觉 | raw.针刺痛觉')

raw['label'] = pd.eval('(raw.upper | raw.lower) & raw.symptom & raw.sign')
raw = raw.set_index('姓名')

names = os.listdir('clean/OCTA')
labels = [raw.loc[name.split('_')[0], 'label'] for name in names]


# label = pd.DataFrame({
#     'name': raw['姓名'].values,
#     'label': raw['label'].astype(int).values,
# })
label = pd.DataFrame({
    'name': names,
    'label': labels,
})
label['label'] = label['label'].astype(int)

print(label['label'].value_counts())


label.to_csv('./clean/label.csv', index=False, encoding='utf-8')
