import matplotlib.pyplot as plt

def draw(df,kw):
    dfg = df.groupby(by='u')
    dfc = dfg.count()['i']
    dfc = dfc.sort_values(ascending=False)
    plt.cla()
    dfc.plot(kind='bar', title= 'ml-1m (%s)'%kw, x='userid',y='#purchase',logy=True)
    plt.savefig('%s.png'%kw)


test=mmread('test.0.mtx')
import pandas as pd
test = test.A
test = test.nonzero()
el = []
for u,i in zip(test[0], test[1]):
    el.append((u,i))
test = pd.DataFrame(el)
