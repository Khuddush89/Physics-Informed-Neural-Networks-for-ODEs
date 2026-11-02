"""Hand calculation of Sec. 4.2 (MLP) and Sec. 4.2.1 (Chebyshev first/second kind, Legendre
features of degree one): forward pass, derivative in t, residual, IC loss, all gradients and one
Adam step for y'=y, y(0)=1, collocation point t1=0.5. Checks manual Adam against torch.optim.Adam."""
import argparse, json, pathlib
import torch
torch.set_default_dtype(torch.float64)
t1=0.5
def run(kind):
    # kind: 'mlp' (7 params), 'cheb1'/'leg' features (1, t), 'cheb2' features (1, 2t)
    if kind=='mlp':
        names=['w1','w2','b1','b2','v1','v2','c']; init=[0.1,0.2,0.1,0.2,0.5,0.6,0.1]
    else:
        names=['w11','w12','w21','w22','b1','b2','v1','v2','c']; init=[0.1,0.2,0.3,0.4,0.1,0.2,0.5,0.6,0.1]
    P=[torch.tensor(x,requires_grad=True) for x in init]
    def net(t):
        if kind=='mlp':
            w1,w2,b1,b2,v1,v2,c=P; z1=w1*t+b1; z2=w2*t+b2
        else:
            w11,w12,w21,w22,b1,b2,v1,v2,c=P
            phi1 = 2*t if kind=='cheb2' else t
            z1=w11*1+w12*phi1+b1; z2=w21*1+w22*phi1+b2
        return v1*torch.tanh(z1)+v2*torch.tanh(z2)+c, (z1,z2)
    def loss():
        t=torch.tensor(t1,requires_grad=True)
        y,(z1,z2)=net(t); dy=torch.autograd.grad(y,t,create_graph=True)[0]
        r=dy-y; y0,_=net(torch.tensor(0.)); Lr=r**2; Li=(y0-1)**2
        return Lr+Li,dict(z=(z1.item(),z2.item()),h=(torch.tanh(z1).item(),torch.tanh(z2).item()),y=y.item(),dy=dy.item(),r=r.item(),y0=y0.item(),Lr=Lr.item(),Li=Li.item(),Lr_t=Lr,Li_t=Li)
    L,info=loss()
    gr=torch.autograd.grad(info['Lr_t'],P,retain_graph=True); gi=torch.autograd.grad(info['Li_t'],P,retain_graph=True)
    g=torch.autograd.grad(L,P)
    opt=torch.optim.Adam(P,lr=1e-3,betas=(0.9,0.999),eps=1e-8)
    for p,gg in zip(P,g): p.grad=gg.clone()
    opt.step()
    # manual adam
    man=[]
    for x,gg in zip(init,g):
        gg=gg.item(); m=0.1*gg; v=0.001*gg*gg; mh=m/0.1; vh=v/0.001; new=x-1e-3*mh/(vh**0.5+1e-8)
        man.append(dict(g=gg,m=m,v=v,mh=mh,vh=vh,new=new))
    L2,_=loss()
    for i,p in enumerate(P): assert abs(p.item()-man[i]['new'])<1e-15
    for k in ['Lr_t','Li_t']: info.pop(k)
    return dict(kind=kind,names=names,init=init,info=info,L=L.item(),gr=[x.item() for x in gr],gi=[x.item() for x in gi],adam=man,L_after=L2.item())
out=[run(k) for k in ['mlp','cheb1','leg','cheb2']]
ap=argparse.ArgumentParser()
ap.add_argument('--quick',action='store_true',help='accepted for the test suite; the calculation is the same')
ap.add_argument('--outdir',default=str(pathlib.Path(__file__).resolve().parent/'output'))
args=ap.parse_args()
od=pathlib.Path(args.outdir); od.mkdir(parents=True,exist_ok=True)
json.dump(out,open(od/'results.json','w'),indent=1)
for o in out:
    print('==',o['kind'],o['info'],'L=%.6g'%o['L'],'L_after=%.6g'%o['L_after'])
    for n,a,b,ad in zip(o['names'],o['gr'],o['gi'],o['adam']):
        print(f"{n:4s} gr={a:.6g} gi={b:.6g} g={ad['g']:.6g} m={ad['m']:.6g} v={ad['v']:.6g} new={ad['new']:.6f}")
