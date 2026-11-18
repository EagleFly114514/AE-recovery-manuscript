# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AE recovery study contributors
"""Clinical figures calculated from runtime inputs."""
from pathlib import Path
import os, json
os.environ['MPLBACKEND']='Agg'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,PathPatch,Rectangle
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

ROOT=Path(os.environ['AE_RUN_ROOT']).resolve();OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
DATA_ROOT=ROOT
COL={'I':'#2A9D8F','D3':'#E9C46A','D45':'#DF8546','X':'#35566C','M':'#E3E8ED'}
BG={'pink':'#F5F7F9','lavender':'#ECF5F3','sage':'#FBF5E5'}
INK='#29333A';EDGE='#35566C';ACCENT='#82658E'
HATCH={'I':'','D3':'//','D45':'\\\\','X':'xx','M':'..'}
plt.rcParams.update({'font.family':['Arial','Microsoft YaHei'],'font.size':8,'text.color':INK,
 'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'axes.titlesize':9,'axes.labelsize':8,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'hatch.linewidth':.3,
 'xtick.major.width':.5,'ytick.major.width':.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})

def save(fig,name):
    for ext in ['png','pdf','svg','tiff']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=600 if ext=='tiff' else 260,facecolor='white',
            **({'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {}))
    plt.close(fig)

def heading(fig,y,letter,text):
    fig.text(.035,y,letter,fontsize=11,weight='bold',va='top')
    fig.text(.065,y,text,fontsize=9.2,weight='bold',va='top')

def box(ax,x,y,w,h,text,color='white',size=8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.35,rounding_size=1.25',
        fc=color,ec='#94A6AC',lw=.7,clip_on=False))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.28)

def arrow(ax,a,b):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>','color':EDGE,'lw':.9,'mutation_scale':8})

def fig1(zh=False):
    frame=pd.read_csv(DATA_ROOT/'data/cohort.csv')
    spec=json.loads((DATA_ROOT/'data/clinical_spec.json').read_text(encoding='utf-8'))
    screening=spec['screening']
    screened=int(screening['internal_screened'])
    treated=screened-int(screening['no_immunotherapy'])-int(screening['infectious_encephalitis'])
    cohorts={name:frame.loc[frame.cohort.eq(name)] for name in ['development','temporal']}
    alive={name:part.loc[part.mrs6.between(0,5)] for name,part in cohorts.items()}
    risk=alive['development'].loc[alive['development'].mrs6.between(3,5)]
    fig=plt.figure(figsize=(7.2,8.2))
    heading(fig,.982,'a','临床问题与名义随访时间点' if zh else 'Clinical questions and nominal follow-up')
    a=fig.add_axes([.045,.775,.91,.175]);a.set(xlim=(0,100),ylim=(0,100));a.axis('off')
    texts=['T0\n记录治疗启动','名义六个月\n按依赖程度分层','名义十二个月\n报告四种状态','名义二十四个月\n连接配对结局'] if zh else [
        'T0\nRecorded treatment','Nominal 6 months\nStratify dependence','Nominal 12 months\nReport four states','Nominal 24 months\nLink paired outcomes']
    for i,(x,label) in enumerate(zip([0,26,52,78],texts)):
        box(a,x,51,22,43,label,[BG['pink'],BG['lavender'],BG['sage'],BG['sage']][i],7.9)
        if i<3:arrow(a,(x+22,73),(x+26,73))
    a.text(0,32,'内部：记录的名义六、十二、二十四个月访视。' if zh else 'Internal assessments: recorded nominal 6-, 12- and 24-month visits.',fontsize=8)
    a.text(0,11,'内部开发及同院时间队列。' if zh else 'Development and same-centre temporal cohorts.',fontsize=8)
    heading(fig,.752,'b','内部队列及固定恢复风险集' if zh else 'Internal cohorts and fixed recovery risk set')
    fig.text(.66,.752,'c',fontsize=11,weight='bold',va='top')
    fig.text(.69,.752,'多中心随访' if zh else 'Multicentre follow-up',fontsize=8.8,weight='bold',va='top')
    b=fig.add_axes([.045,.147,.575,.575]);c=fig.add_axes([.665,.147,.29,.575])
    for ax in [b,c]:ax.set(xlim=(-2,102),ylim=(-8,100));ax.axis('off')
    box(b,0,91,100,8,f'筛查登记 n = {screened}' if zh else f'Screening registry  n = {screened}',BG['pink'])
    arrow(b,(25,91),(25,79))
    excluded=f"No immunotherapy {screening['no_immunotherapy']}; infectious encephalitis {screening['infectious_encephalitis']}"
    b.text(33,84.8,excluded,fontsize=7.2,va='center')
    box(b,0,71,100,8,f'接受治疗 n = {treated}' if zh else f'Treated cohort  n = {treated}',BG['pink'])
    arrow(b,(25,71),(25,56))
    excluded=f"Age <14: {screening['below_age_14']}; timeline: {screening['inconsistent_internal_timeline']}\nQualifying treatment date unavailable: {screening['missing_qualifying_treatment_date']}"
    b.text(33,64,excluded,fontsize=7.2,va='center')
    box(b,0,48,100,8,f'临床确认的内部队列 n = {len(frame)}' if zh else f'Clinically confirmed internal cohort  n = {len(frame)}',BG['lavender'],7.8)
    for name,x,width in [('development',0,55),('temporal',65,35)]:
        part=cohorts[name];living=alive[name]
        arrow(b,(x+width/2,48),(x+width/2,40))
        label='Development' if name=='development' else 'Temporal'
        box(b,x,30,width,10,f'{label}: {len(part)}',BG['lavender'],7.8)
        arrow(b,(x+width/2,30),(x+width/2,18))
        b.text(x+1,24,f'Death {part.mrs6.eq(6).sum()}; missing {part.mrs6.isna().sum()}',fontsize=7.0,ha='left',bbox=dict(fc='white',ec='none',pad=.5))
        box(b,x,8,width,10,f'Living, assessed: {len(living)}\nDependent: {living.mrs6.between(3,5).sum()}',BG['sage'],7.4)
    arrow(b,(27,8),(27,-.5))
    box(b,0,-8,100,7.3,f'Fixed recovery risk set: {len(risk)}  |  mRS 3: {risk.mrs6.eq(3).sum()}; mRS 4–5: {risk.mrs6.ge(4).sum()}',BG['sage'],7.5)
    metadata=DATA_ROOT/'data/multicentre_collection.json'
    if metadata.is_file():
        progress=json.loads(metadata.read_text(encoding='utf-8'))
        collection=f"{progress['external_hospitals']} external hospitals\n{progress['registry_records']} registry records"
    else:
        collection='Prospective follow-up\nSeparate evaluation cohort'
    box(c,0,79,100,20,collection,BG['lavender'],7.8)
    c.text(50,71,'随访评价' if zh else 'Follow-up evaluation',ha='center',fontsize=7.6,weight='bold')
    planned=[(48,'Confirm diagnoses\nand index episodes'),(25,'Verify dated\n6-, 12- and 24-month visits'),(2,'Replicate recovery patterns\nand evaluate frozen models')]
    for y,label in planned:
        box(c,0,y,100,14,label,'white',7.5);c.patches[-1].set_linestyle((0,(3,2)))
    for top,bottom in [(67,62),(48,39),(25,16)]:
        c.annotate('',xy=(50,bottom),xytext=(50,top),arrowprops={'arrowstyle':'-|>','color':EDGE,'lw':.8,'linestyle':'--','mutation_scale':8})
    c.text(50,-6,'虚线：计划步骤' if zh else 'Dashed boxes: planned steps',ha='center',fontsize=7,color='#697780')
    foot=fig.add_axes([.045,.020,.91,.107]);foot.set(xlim=(0,100),ylim=(0,100));foot.axis('off')
    foot.text(0,92,f'Recovery states known: 12 months {risk.mrs12.notna().sum()}/{len(risk)}; 24 months {risk.mrs24.notna().sum()}/{len(risk)}.',fontsize=7.9)
    observed={name:part.loc[part.mrs12.notna()] for name,part in alive.items()}
    foot.text(0,71,'Auxiliary prediction: '+'; '.join(f'{name} {len(part)} ({part.mrs12.ge(3).sum()} events)' for name,part in observed.items())+'.',fontsize=7.7)
    box(foot,0,12,43,37,'Development-only selection, tuning and fit',BG['lavender'],7.7)
    box(foot,50,12,15,37,'Freeze',BG['pink'],8.1)
    box(foot,72,12,28,37,'Same-centre temporal evaluation',BG['sage'],7.2)
    arrow(foot,(43,30),(50,30));arrow(foot,(65,30),(72,30))
    save(fig,'Figure_1'+('_ZH' if zh else ''))

def flow_summary():
    transitions=pd.read_csv(DATA_ROOT/'results/longitudinal_transition_counts.csv')
    first=transitions.loc[transitions.interval.eq('worst_to_six')]
    final=transitions.loc[transitions.interval.eq('twelve_to_twentyfour')]
    late=final.loc[final.from_state.isin(['D3','D45'])&final.to_state.eq('I'),'n'].sum()
    states=pd.read_csv(DATA_ROOT/'results/clinical_four_state.csv')
    risk=states.loc[states.population.eq('Primary')&states.month.eq(12)&states.stratum.eq('3-5'),'n'].iloc[0]
    paired=pd.read_csv(DATA_ROOT/'figures/source_paired_by_prior_severity.csv')
    return int(first.n.sum()),int(late),int(paired.iloc[-1].independent),int(risk)

def sankey(ax,zh):
    data=pd.read_csv(DATA_ROOT/'results/longitudinal_transition_counts.csv')
    states=['M','I','D3','D45','X'];intervals=['worst_to_six','six_to_twelve','twelve_to_twentyfour']
    mats=[data.loc[data.interval.eq(k)].pivot(index='from_state',columns='to_state',values='n').loc[states,states].to_numpy() for k in intervals]
    counts=[mats[0].sum(1),*[m.sum(0) for m in mats]];n=int(mats[0].sum());gap=15.;xs=[0,1,2,3];width=.085
    assert all(sum(c)==n for c in counts)
    bottoms=[]
    for cnt in counts:
        top=n+4*gap;pos=[]
        for value in cnt:pos.append(top-value);top-=value+gap
        bottoms.append(np.array(pos))
    highlight=[]
    for k,m in enumerate(mats):
        left=bottoms[k].copy()+counts[k];right=bottoms[k+1].copy()+counts[k+1]
        for i in range(5):
            for j in range(5):
                value=m[i,j]
                if not value:continue
                x0=xs[k]+width;x1=xs[k+1];y0=left[i]-value;y1=right[j]-value;mid=(x0+x1)/2
                vertices=[(x0,y0),(mid,y0),(mid,y1),(x1,y1),(x1,y1+value),(mid,y1+value),(mid,y0+value),(x0,y0+value),(x0,y0)]
                codes=[MPath.MOVETO,*[MPath.CURVE4]*3,MPath.LINETO,*[MPath.CURVE4]*3,MPath.CLOSEPOLY]
                late=k==2 and i in [2,3] and j==1
                missing=states[i]=='M' or states[j]=='M'
                colour=COL['M'] if missing else (ACCENT if late else COL[states[i]])
                p=PathPatch(MPath(vertices,codes),facecolor=colour,edgecolor='none',alpha=.92 if late else (.35 if missing else .38),zorder=3 if late else 1)
                ax.add_patch(p)
                if late:highlight.append([int(value),float(y0),float(y1)])
                left[i]-=value;right[j]-=value
    for k,cnt in enumerate(counts):
        text_y=bottoms[k]+cnt/2
        for i in range(1,5):
            if cnt[i] and cnt[i-1]:text_y[i]=min(text_y[i],text_y[i-1]-27)
        for i,value in enumerate(cnt):
            if not value:continue
            ax.add_patch(Rectangle((xs[k],bottoms[k][i]),width,value,fc=COL[states[i]],ec='white',lw=.75,hatch=HATCH[states[i]],zorder=4))
            if abs(text_y[i]-(bottoms[k][i]+value/2))>1:
                ax.plot([xs[k],xs[k]-.025],[bottoms[k][i]+value/2,text_y[i]],color=EDGE,lw=.4)
            ax.text(xs[k]-.035,text_y[i],str(int(value)),va='center',ha='right',fontsize=7.6,color=INK,weight='bold',
                bbox=dict(facecolor='white',edgecolor='none',alpha=.88,pad=.4),zorder=5)
    labels=['住院最差','6个月','12个月','24个月'] if zh else ['Worst inpatient','6 months','12 months','24 months']
    for x,label in zip(xs,labels):ax.text(x+width/2,n+4*gap+34,label,ha='center',fontsize=7.8,weight='bold')
    ax.set(xlim=(-1.12,3.33),ylim=(-10,n+4*gap+64));ax.axis('off')
    ax.text(-1.08,620,'各次访视状态' if zh else 'States at each visit',fontsize=7.5,weight='bold')
    names=['缺失','独立\nmRS 0–2','中度依赖\nmRS 3','重度依赖\nmRS 4–5','死亡'] if zh else ['Missing','Independent\nmRS 0–2','Moderate\nmRS 3','Severe\nmRS 4–5','Death']
    for y,key,label in zip([530,423,316,209,102],states,names):
        ax.add_patch(Rectangle((-1.08,y-15),.105,30,fc=COL[key],ec='#6D7881',lw=.55,hatch=HATCH[key]))
        ax.text(-.92,y,label,va='center',fontsize=7.4,linespacing=1.12)
    return counts

def stacked(ax,values,labels,zh):
    keys=['I','D3','X','M'];ys=np.arange(len(values))[::-1]
    for y,vals in zip(ys,values):
        total=sum(vals);left=0
        for value,key in zip(vals,keys):
            w=value/total*100
            if value:
                ax.barh(y,w,left=left,color=COL[key],height=.54,edgecolor=EDGE,linewidth=.4,hatch=HATCH[key])
                ax.text(left+w/2,y,str(int(value)),va='center',ha='center',fontsize=8,color='white' if key=='X' else INK,weight='bold')
            left+=w
        ax.text(103,y,f'{vals[0]}/{total} ({100*vals[0]/total:.1f}%)',va='center',fontsize=7.7)
    ax.set(xlim=(0,133),ylim=(-.6,len(values)-.4),yticks=ys,yticklabels=labels,xticks=[0,25,50,75,100],xlabel='各行固定人群的比例（%）' if zh else 'Percentage of each fixed row population')
    ax.spines['bottom'].set_bounds(0,100);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,labelsize=7.8)
    ax.text(103,len(values)-.32,'独立 / 全部' if zh else 'Independent / all',va='bottom',fontsize=7.4)
    names=['独立','仍依赖 (3–5)','死亡','缺失'] if zh else ['Independent','Dependent (3–5)','Death','Missing']
    return [Rectangle((0,0),1,1,fc=COL[k],ec=EDGE,lw=.4,hatch=HATCH[k]) for k in keys],names

def paired_by_prior():
    r=pd.read_csv(DATA_ROOT/'figures/source_paired_by_prior_severity.csv')
    r.to_csv(OUT/'source_paired_by_prior_severity.csv',index=False)
    assert r[['independent','dependent','death','missing']].sum(axis=1).eq(r['n']).all()
    return r

def fig2(zh=False):
    fig=plt.figure(figsize=(7.2,8.25))
    heading(fig,.982,'a','六个月依赖程度区分后续独立' if zh else 'Subsequent independence by six-month dependence severity')
    a=fig.add_axes([.22,.765,.74,.17]);f=pd.read_csv(DATA_ROOT/'results/clinical_four_state.csv');f=f.loc[f.population.eq('Primary')]
    sizes=f.loc[f.month.eq(12)].set_index('stratum')['n']
    total,full_late,primary_late,primary_n=flow_summary()
    source_rows=[]
    for month,dy,marker,color,fill in [(12,.16,'o',EDGE,COL['D3']),(24,-.16,'s',ACCENT,COL['D45'])]:
        s=f.loc[f.month.eq(month)].set_index('stratum').loc[['3','4-5','3-5']]
        yy=np.array([2,1,0])+dy;p=s.p.to_numpy()*100
        a.errorbar(p,yy,xerr=[p-100*s.lower,100*s.upper-p],fmt=marker,ms=4.4,mfc=fill,mec=color,capsize=2,color=color,lw=.95)
        for y,r in zip(yy,s.itertuples()):
            counts=f'{r.independent}/{r.observed}'
            percentage=f'{100*r.p:.1f} ({100*r.lower:.1f}–{100*r.upper:.1f})'
            a.text(108,y,counts,fontsize=7.7,va='center',color=color)
            a.text(164,y,percentage,fontsize=7.7,va='center',color=color)
            source_rows.append({'month':month,'stratum':r.Index,'independent':r.independent,'known':r.observed,
                'independent_pct':100*r.p,'ci_lower_pct':100*r.lower,'ci_upper_pct':100*r.upper,
                'count_label':counts,'percentage_ci_label':percentage})
    a.set(xlim=(0,230),ylim=(-.55,2.55),xticks=[0,20,40,60,80,100],yticks=[2,1,0],yticklabels=[f"mRS 3 (n = {int(sizes['3'])})",f"mRS 4–5 (n = {int(sizes['4-5'])})",f"All dependent (n = {int(sizes['3-5'])})"],xlabel='后续独立（%）' if zh else 'Subsequent independence (%)')
    a.xaxis.set_label_coords(50/230,-.22)
    a.spines['bottom'].set_bounds(0,100);a.spines['left'].set_visible(False);a.tick_params(axis='y',length=0,labelsize=8)
    a.text(108,2.6,'独立 / 已知结局' if zh else 'Independent / known',fontsize=7.1,va='bottom')
    a.text(164,2.6,'百分比（95% CI）' if zh else '% (95% CI)',fontsize=7.3,va='bottom')
    fig.legend([Line2D([],[],marker='o',color=EDGE,mfc=COL['D3'],lw=.9,markersize=4),Line2D([],[],marker='s',color=ACCENT,mfc=COL['D45'],lw=.9,markersize=4)],
        ['12个月','24个月'] if zh else ['12 months','24 months'],loc='upper left',bbox_to_anchor=(.215,.963),frameon=False,ncol=2,fontsize=7.8)
    heading(fig,.708,'b','从住院最差状态至24个月的功能转移' if zh else 'Functional transitions from inpatient illness to 24 months')
    fig.text(.065,.679,f'完整开发队列 · 固定分母 n = {total} · 带宽 = 患者人数' if zh else f'Full development cohort · fixed denominator n = {total} · ribbon width = patients',color='#697780',fontsize=7.6)
    b=fig.add_axes([.030,.315,.945,.345]);sankey(b,zh)
    fig.text(.065,.281,f'Purple: {full_late} later independence transitions; {primary_late} belong to the primary {primary_n}-person risk set.',color=ACCENT,fontsize=7.6,weight='bold')
    heading(fig,.236,'c','十二个月仍依赖者：保留其六个月状态' if zh else 'Still dependent at 12 months: retain the earlier six-month state')
    c=fig.add_axes([.25,.090,.71,.108]);r=paired_by_prior()
    labels=[f'Prior mRS 3 (n = {int(r.iloc[0]["n"])})',f'Prior mRS 4–5 (n = {int(r.iloc[1]["n"])})',f'All 12-month dependent ({int(r.iloc[2]["n"])})']
    handles,names=stacked(c,r[['independent','dependent','death','missing']].values.tolist(),labels,zh)
    fig.legend(handles,names,loc='lower center',bbox_to_anchor=(.56,.001),ncol=4,frameon=False,fontsize=7.6)
    pd.DataFrame(source_rows).to_csv(OUT/'source_Figure_2a.csv',index=False)
    fig.savefig(OUT/('Figure_2a'+('_ZH' if zh else '')+'.png'),dpi=260,facecolor='white',bbox_inches=Bbox.from_extents(0,.716*8.25,7.2,8.25))
    save(fig,'Figure_2'+('_ZH' if zh else ''))

def fig3(zh=False):
    fig=plt.figure(figsize=(7.2,6.1));f=pd.read_csv(DATA_ROOT/'results/clinical_contrasts.csv')
    states=pd.read_csv(DATA_ROOT/'results/clinical_four_state.csv')
    keys=['Primary','Multiple episodes retained','NMDAR positive','Neuronal-surface antibody subset']
    sizes=f.loc[f.month.eq(12)].set_index('population')['n']
    labels=['Primary','Episodes retained','NMDAR positive','Surface antibodies']
    names=[f'{label} (n = {int(sizes[key])})' for key,label in zip(keys,labels)]
    heading(fig,.978,'a','独立比例差：六个月 mRS 3 减 mRS 4–5' if zh else 'Independence difference: six-month mRS 3 minus mRS 4–5')
    source_rows=[]
    for month,x in [(12,.265),(24,.645)]:
        fig.text(x,.913,('%d个月'%month) if zh else ('%d months'%month),fontsize=9.2,weight='bold',va='top')
        primary=states.loc[states.population.eq('Primary') & states.month.eq(month) & states.stratum.eq('3-5')].iloc[0]
        fig.text(x,.871,('主分析缺失：%d/%d'%(primary['missing'],primary['n'])) if zh else ('Primary missing: %d/%d'%(primary['missing'],primary['n'])),fontsize=7.2,color='#697780')
        ax=fig.add_axes([x,.565,.335,.280]);s=f.loc[f.month.eq(month)].set_index('population').loc[keys]
        for yy,r in zip([3,2,1,0],s.itertuples()):
            ax.axhline(yy,color='#F0F1F3',lw=.6,zorder=0)
            ax.plot(100*np.array([r.fixed_lower,r.fixed_upper]),[yy-.18]*2,color='#B3AEB5',lw=3.6,solid_capstyle='butt',zorder=2)
            ax.errorbar(100*r.difference,yy+.18,xerr=[[100*(r.difference-r.lower)],[100*(r.upper-r.difference)]],fmt='o' if month==12 else 's',
                color=EDGE if month==12 else ACCENT,mfc=COL['D3'] if month==12 else COL['D45'],ms=4.0,capsize=2,lw=1,zorder=3)
            estimate_label=f'{100*r.difference:.1f} ({100*r.lower:.1f}–{100*r.upper:.1f})'
            bounds_label=f'{100*r.fixed_lower:.1f}–{100*r.fixed_upper:.1f}'
            ax.text(94,yy+.18,estimate_label,fontsize=7.1,va='center',color=EDGE if month==12 else ACCENT)
            ax.text(94,yy-.18,bounds_label,fontsize=7.1,va='center',color='#746B79')
            source_rows.append({'population':r.Index,'month':month,'n':r.n,'risk_difference_pp':100*r.difference,
                'ci_lower_pp':100*r.lower,'ci_upper_pp':100*r.upper,'missing_bound_lower_pp':100*r.fixed_lower,
                'missing_bound_upper_pp':100*r.fixed_upper,'estimate_ci_label':estimate_label,'missing_bounds_label':bounds_label})
        ax.axvline(0,c=EDGE,ls=(0,(3,3)),lw=.75,zorder=1)
        ax.text(94,3.66,'差值（95% CI）' if zh else 'RD (95% CI)',fontsize=6.9,va='bottom')
        ax.set(xlim=(-5,180),ylim=(-.60,3.70),xticks=[0,20,40,60,80],yticks=[3,2,1,0],yticklabels=names if month==12 else ['']*4)
        ax.spines['bottom'].set_bounds(0,80)
        ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,labelsize=7.9,pad=8)
        ax.tick_params(axis='x',labelsize=7.6)
    fig.text(.55,.505,'风险差（百分点）' if zh else 'Risk difference (percentage points)',fontsize=8,ha='center')
    interval_handles=[Line2D([],[],marker='o',color=EDGE,mfc=COL['D3'],lw=1,markersize=4),Line2D([],[],color='#B3AEB5',lw=3.6,solid_capstyle='butt')]
    fig.legend(interval_handles,['观察差值与95% CI','缺失结局界限'] if zh else ['Observed difference and 95% CI','Missing-outcome bounds'],
        loc='center',bbox_to_anchor=(.55,.464),frameon=False,ncol=2,fontsize=7.1,handlelength=2.1,columnspacing=1.8)
    heading(fig,.405,'b','十二个月依赖者的二十四个月配对结局' if zh else '24-month outcomes after dependence at 12 months')
    c=fig.add_axes([.265,.150,.68,.200]);r=pd.read_csv(DATA_ROOT/'results/clinical_paired_sensitivity.csv').set_index('population').loc[keys]
    rowlabels=[f'{label} (n = {int(r.loc[key,"twelve_dependent"])})' for key,label in zip(keys,labels)]
    handles,names4=stacked(c,r[['independent','dependent','death','missing']].values.tolist(),rowlabels,zh)
    c.tick_params(axis='y',pad=8)
    c.set_xlabel('各行人群的比例（%）' if zh else 'Percentage of each population',labelpad=5)
    fig.legend(handles,names4,loc='lower center',bbox_to_anchor=(.60,.025),ncol=4,frameon=False,fontsize=7.4)
    pd.DataFrame(source_rows).to_csv(OUT/'source_Figure_3_contrasts.csv',index=False)
    r[['independent','dependent','death','missing']].to_csv(OUT/'source_Figure_3_paired.csv')
    save(fig,'Figure_3'+('_ZH' if zh else ''))

if __name__=='__main__':
    fig1(False);fig2(False);fig3(False)
    total,_,_,_=flow_summary()
    standalone=plt.figure(figsize=(9.4,4.3))
    heading(standalone,.98,'b','Functional transitions from inpatient illness to 24 months')
    standalone.text(.065,.905,f'Full development cohort · fixed denominator n = {total} · ribbon width = patients',fontsize=9,color='#697780')
    ax=standalone.add_axes([.03,.065,.945,.79]);sankey(ax,False)
    save(standalone,'Figure_2b')
    (OUT/'palette.json').write_text(json.dumps(dict(states=COL,node_order_top_to_bottom=['M','I','D3','D45','X'],panel_backgrounds=BG,late_transition=ACCENT),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Figures exported from runtime inputs; missing state uppermost.')
