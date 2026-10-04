"""Does GENERIC square-attention discover ray structure on its own?

Trains a 2-layer, 4-head self-attention readout over the 64 board squares to
predict legal moves from the TRUE board, then measures what fraction of its
attention mass lands on ray-connected (collinear) square pairs.

The null is exact: 1456 of the 4032 ordered off-diagonal pairs are
ray-connected, so uniform attention puts 36.1% of its mass on rays.  Well above
that means training finds the structure unaided -- and a ray MASK would be
largely redundant.
"""
import glob, os, pickle, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.othello import OthelloBoardState
from train_board_state_gpt import CELL_TO_TOK

CENTER={27,28,35,36}; VALID=[c for c in range(64) if c not in CENTER]
DIRS=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

def ray_mask():
    M=torch.zeros(64,64,dtype=torch.bool)
    for s in range(64):
        r,c=s//8,s%8
        for dr,dc in DIRS:
            rr,cc=r+dr,c+dc
            while 0<=rr<8 and 0<=cc<8:
                M[s,rr*8+cc]=True; rr+=dr; cc+=dc
    return M
RAY=ray_mask()

def collect(files, n_pos):
    B,L=[],[]
    for f in files:
        for g in pickle.load(open(f,'rb')):
            if len(g)<60: continue
            bd=OthelloBoardState()
            for ply in range(54):
                if ply>=5:
                    lg=bd.get_valid_moves()
                    if lg:
                        st=bd.state.flatten(); mv=bd.next_hand_color
                        lab=np.zeros(64,np.int8); lab[st==-mv]=1; lab[st==mv]=2
                        m=np.zeros(60,bool)
                        for c in lg: m[CELL_TO_TOK[c]-1]=True
                        B.append(lab); L.append(m)
                bd.update([g[ply]])
            if len(B)>=n_pos: break
        if len(B)>=n_pos: break
    return np.stack(B[:n_pos]), np.stack(L[:n_pos])

class Attn(nn.Module):
    """Explicit attention so the weights can be read out."""
    def __init__(s, d=96, heads=4):
        super().__init__(); s.h=heads; s.dk=d//heads
        s.q=nn.Linear(d,d); s.k=nn.Linear(d,d); s.v=nn.Linear(d,d); s.o=nn.Linear(d,d)
    def forward(s, x, keep=False):
        B,N,D=x.shape
        q=s.q(x).view(B,N,s.h,s.dk).transpose(1,2)
        k=s.k(x).view(B,N,s.h,s.dk).transpose(1,2)
        v=s.v(x).view(B,N,s.h,s.dk).transpose(1,2)
        a=torch.softmax(q@k.transpose(-2,-1)/s.dk**0.5, -1)
        s.last = a.detach() if keep else None
        return s.o((a@v).transpose(1,2).reshape(B,N,D))

class SquareAttn(nn.Module):
    def __init__(s, d=96, layers=2, heads=4):
        super().__init__()
        s.emb=nn.Embedding(3,d); s.row=nn.Embedding(8,d); s.col=nn.Embedding(8,d)
        s.at=nn.ModuleList([Attn(d,heads) for _ in range(layers)])
        s.ln1=nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        s.ln2=nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        s.mlp=nn.ModuleList([nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
                             for _ in range(layers)])
        s.out=nn.Linear(d,1)
        s.register_buffer('rid',torch.arange(64)//8); s.register_buffer('cid',torch.arange(64)%8)
        s.register_buffer('valid',torch.tensor(VALID))
    def forward(s, b, keep=False):
        x=s.emb(b)+s.row(s.rid)+s.col(s.cid)
        for i,at in enumerate(s.at):
            x=x+at(s.ln1[i](x), keep=keep); x=x+s.mlp[i](s.ln2[i](x))
        return s.out(x).squeeze(-1)[:, s.valid]

def soft_ce(lg, L):
    q=L.float(); q=q/q.sum(1,keepdim=True)
    return -(q*F.log_softmax(lg,1)).sum(1).mean()

if __name__=='__main__':
    fs=sorted(glob.glob('data/othello_synthetic/*.pickle'))
    t0=time.time()
    Btr,Ltr=collect(fs[:4], 150000); Bte,Lte=collect(fs[-3:], 20000)
    print('train %s  test %s  (%.0fs)'%(Btr.shape,Bte.shape,time.time()-t0), flush=True)
    dev='mps' if torch.backends.mps.is_available() else 'cpu'
    m=SquareAttn().to(dev)
    print('%d params, device %s'%(sum(p.numel() for p in m.parameters()),dev), flush=True)
    opt=torch.optim.AdamW(m.parameters(), lr=2e-3)
    EP,BS=8,512
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=2e-3,
         total_steps=EP*((len(Btr)+BS-1)//BS), pct_start=0.1)
    xb=torch.from_numpy(Btr).long(); yb=torch.from_numpy(Ltr)
    for ep in range(EP):
        perm=torch.randperm(len(xb))
        for i in range(0,len(xb),BS):
            j=perm[i:i+BS]
            loss=soft_ce(m(xb[j].to(dev)), yb[j].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        print('  ep%d loss %.4f (%.0fs)'%(ep,loss.item(),time.time()-t0), flush=True)
    # accuracy
    m.eval(); accs=[]
    with torch.no_grad():
        lo=[]
        for i in range(0,len(Bte),BS):
            lo.append(m(torch.from_numpy(Bte[i:i+BS]).long().to(dev)).cpu())
        lo=torch.cat(lo)
    Lt=torch.from_numpy(Lte)
    p=torch.softmax(lo.float(),1); ill=(p*~Lt).sum(1)
    top1=(torch.gather(Lt,1,lo.argmax(1,keepdim=True)).float().mean())
    print('\nheld-out: top-1 %.2f%%  median illegal mass %.3f%%'
          %(100*top1, 100*ill.median()), flush=True)
    # ---- the measurement ----
    with torch.no_grad():
        m(torch.from_numpy(Bte[:512]).long().to(dev), keep=True)
    off=~torch.eye(64,dtype=torch.bool)
    null=RAY[off].float().mean().item()
    print('\nNULL: %.1f%% of off-diagonal pairs are ray-connected' % (100*null))
    print('attention mass on RAY pairs (self-attention excluded, renormalised):')
    for li,at in enumerate(m.at):
        a=at.last.cpu().float()                       # (B, h, 64, 64)
        a=a*off                                        # drop the diagonal
        a=a/a.sum(-1,keepdim=True).clamp_min(1e-9)
        for h in range(a.shape[1]):
            frac=(a[:,h]*RAY).sum(-1).mean().item()
            print('  layer %d head %d:  %5.1f%%   (%+.1f pp vs null)'
                  % (li,h,100*frac,100*(frac-null)))
        tot=(a*RAY).sum(-1).mean().item()
        print('  layer %d MEAN     :  %5.1f%%   (%+.1f pp vs null)'
              % (li,100*tot,100*(tot-null)))
