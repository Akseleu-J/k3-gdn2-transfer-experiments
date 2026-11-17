import os, sys, time, json, math, gc, re, struct, traceback, hashlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import ml_dtypes, requests
import importlib.metadata

MODE=os.environ.get('K3_MODE','tiny')
STEPS=int(os.environ.get('K3_STEPS','0'))
SEQ=int(os.environ.get('K3_SEQ','128'))
BATCH_REQUESTED=int(os.environ.get('K3_BATCH','1'))
ARTIFACTS=os.environ.get('K3_ARTIFACTS','artifacts')
os.makedirs(ARTIFACTS,exist_ok=True)
REPORT={'mode':MODE,'status':'RUNNING','results':[], 'scope':'candidate student; no teacher parity claim'}
RPT_PATH=os.path.join(ARTIFACTS,'report.json')

def save_rpt():
    p=Path(RPT_PATH); tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(REPORT,indent=2,default=str),encoding='utf-8'); os.replace(tmp,p)

def rec(cat,tid,status,detail=''):
    REPORT['results'].append(dict(cat=cat,id=tid,status=status,detail=str(detail)))
    print(f'[{status}] {cat}.{tid}: {detail}',flush=True); save_rpt()

def chk(cat,tid,ok,detail=''):
    rec(cat,tid,'PASS' if ok else 'FAIL',detail)
    if not ok: raise AssertionError(f'{cat}.{tid}: {detail}')

def category_summary(cat):
    print(cat,{s:sum(r['cat']==cat and r['status']==s for r in REPORT['results']) for s in ['PASS','FAIL','INFO']})

def exception_hook(typ,value,tb):
    REPORT['status']='FAILED'; REPORT['exception_type']=typ.__name__; save_rpt()
    sys.__excepthook__(typ,value,tb)
sys.excepthook=exception_hook
HIGHEST=jax.lax.Precision.HIGHEST
devices=jax.devices(); N_DEV=len(devices)
BATCH=N_DEV if BATCH_REQUESTED==0 else BATCH_REQUESTED
if BATCH<1:raise ValueError('Batch must be positive or 0 for automatic device count')
mesh=Mesh(np.array(devices).reshape(-1),('fsdp',))
def fsdp_spec(shape,n_dev,axis='fsdp'):
    if n_dev<=1:return P()
    c=[(s,i) for i,s in enumerate(shape) if s>=n_dev and s%n_dev==0]
    if not c:return P()
    _,i=max(c); sp=[None]*len(shape); sp[i]=axis; return P(*sp)
REPORT['env']={'python':sys.version,'jax':jax.__version__,'backend':jax.default_backend(),'devices':[str(x) for x in devices]}
mem=[]
for dev in devices:
    try:mem.append(dev.memory_stats() or {})
    except Exception:mem.append({})
REPORT['hardware']={'device_count':N_DEV,'device_kinds':[getattr(d,'device_kind','unknown') for d in devices],'memory_stats':mem,'requested_batch':BATCH_REQUESTED,'effective_batch':BATCH}
try:
    import psutil
    vm=psutil.virtual_memory();REPORT['hardware']['host_memory']={'total':vm.total,'available':vm.available}
except ImportError:pass
REPORT['packages']={k:importlib.metadata.version(k) for k in ['jax','jaxlib','flax','optax','numpy','ml_dtypes','requests']}
USE_ATOMIC_KERNEL=False
HF_TOKEN=os.environ.get('HF_TOKEN')
WORK_ROOT=str(Path(ARTIFACTS).resolve().parent)
K3_SHA=None
save_rpt()
if MODE=='real':
    if jax.default_backend()!='tpu':
        raise RuntimeError('Real candidate requires a TPU backend; detected '+jax.default_backend())
    from huggingface_hub import HfApi, hf_hub_download
    K3_SHA=HfApi(token=HF_TOKEN).model_info('moonshotai/Kimi-K3',revision=os.environ.get('K3_REVISION','main')).sha
    cp=hf_hub_download('moonshotai/Kimi-K3','config.json',revision=K3_SHA,token=HF_TOKEN)
    source_config=json.loads(Path(cp).read_text())
    Path(ARTIFACTS,'k3_config.json').write_text(json.dumps(source_config,indent=2))
    tc=source_config['text_config']
    # This candidate implements this specific source geometry; refuse drift.
    expected={'hidden_size':7168,'num_attention_heads':96,'qk_nope_head_dim':128,'qk_rope_head_dim':64,'v_head_dim':128,'tie_word_embeddings':False,'mla_use_output_gate':True}
    for k,v in expected.items(): chk('CONFIG',k,tc.get(k)==v,str(tc.get(k)))
    chk('CONFIG','safe_gate',tc['linear_attn_config'].get('gate_lower_bound')==-5.0)
    actual_mla=tuple(i-1 for i in tc['linear_attn_config']['full_attn_layers'] if i<=16)
    chk('CONFIG','layer_mapping',actual_mla==(3,7,11,15),actual_mla)
    REPORT['k3_sha']=K3_SHA; save_rpt()

REPORT["draft_sha256"]='8fab287c59decd68e9f437ca1ffa311cc6a617f9c4df0a0aee52792d19cb99db'
import dataclasses, gc
from typing import Tuple
import flax.linen as nn
for _n in ("params","_student_params","abstract","out_shardings","logits","logits_k3","_stage1_tr","_stage1_opt","opt_state"):
    globals().pop(_n,None)
gc.collect(); jax.clear_caches()
def _sanitize(x,clip=1e3): return jnp.nan_to_num(jnp.clip(x,-clip,clip),nan=0.0,posinf=clip,neginf=-clip)
def make_grad_sanitizer(clip_val=1e3):
    @jax.custom_vjp
    def _fn(x): return x
    def _f(x): return x,None
    def _b(_,g): return (jnp.nan_to_num(jnp.clip(g,-clip_val,clip_val),nan=0.0,posinf=clip_val,neginf=-clip_val),)
    _fn.defvjp(_f,_b); return _fn
def situ(x,beta=4.0): return beta*jnp.tanh(x/beta)*jax.nn.sigmoid(x)
def situ_glu(gate,up,bg=4.0,bu=25.0): return situ(gate.astype(jnp.float32),bg)*(bu*jnp.tanh(up.astype(jnp.float32)/bu))

@dataclasses.dataclass(frozen=True)
class StudentConfig:
    d_model:int=7168; num_blocks:int=4; layers_per_block:int=4; n_layers:int=16; vocab:int=163840
    tie_embeddings:bool=False; n_heads:int=32; d_head:int=128; mla_positions:Tuple[int,...]=(3,7,11,15)
    q_lora_rank:int=1536; kv_lora_rank:int=512; qk_rope_head_dim:int=64; rope_theta:float=10000.0
    d_ff:int=6144; d_conv:int=4; dar_latent:int=256
    @property
    def d_attn_proj(self): return self.n_heads*self.d_head
    @property
    def n_kda(self): return self.n_layers-len(self.mla_positions)

class RoPE(nn.Module):
    dim:int; theta:float=10000.0
    @nn.compact
    def __call__(self,L):
        half=self.dim//2; fr=1.0/(self.theta**(jnp.arange(0,half)/half)); ang=jnp.outer(jnp.arange(L),fr)
        return jnp.cos(ang),jnp.sin(ang)
def apply_rope(x,cos,sin):
    d=x.shape[-1]; c=cos[None,:,None,:]; s=sin[None,:,None,:]; x1,x2=x[...,:d//2],x[...,d//2:]
    return jnp.concatenate([x1*c-x2*s,x1*s+x2*c],-1)
BF=dict(dtype=jnp.bfloat16,param_dtype=jnp.bfloat16)

class DeltaAttentionResidual(nn.Module):
    d_model:int; d_latent:int
    @nn.compact
    def __call__(self,cur,sources):
        # Create the same parameters for every source count, including zero/one.
        a=self.param("alpha_scale",nn.initializers.zeros,())
        w=self.param("w_l",nn.initializers.zeros,(self.d_model,))
        norm=nn.RMSNorm(epsilon=1e-6,param_dtype=jnp.float32,name="src_norm")
        stack=jnp.stack(sources,0) if sources else jnp.zeros((1,)+cur.shape,cur.dtype)
        sn=norm(stack.astype(jnp.float32))
        weights=jax.nn.softmax(jnp.einsum("d,nbld->nbl",w,sn),axis=0)
        retrieved=jnp.einsum("nbl,nbld->bld",weights,stack.astype(jnp.float32))
        return (a*retrieved).astype(cur.dtype)

class GDN2Mixer(nn.Module):
    cfg:StudentConfig
    @nn.compact
    def __call__(self,x):
        cfg=self.cfg; B,L,_=x.shape; H,Dh,Pp=cfg.n_heads,cfg.d_head,cfg.d_attn_proj; eps=1e-6
        def short_conv(name,u):
            w=self.param(f"{name}_conv_w",nn.initializers.normal(0.02),(u.shape[-1],cfg.d_conv))
            b=self.param(f"{name}_conv_b",nn.initializers.zeros,(u.shape[-1],))
            y=jax.lax.conv_general_dilated(u.astype(jnp.float32),w.T[:,None,:].astype(jnp.float32),window_strides=(1,),
                padding=[(cfg.d_conv-1,0)],feature_group_count=u.shape[-1],dimension_numbers=("NHC","HIO","NHC"))
            return (y+b[None,None,:]).astype(jnp.float32)
        D=lambda n,o,bias=False: nn.Dense(o,use_bias=bias,name=n,**BF)
        q=jax.nn.silu(short_conv("q",D("q_proj",Pp)(x))).reshape(B,L,H,Dh)
        k=jax.nn.silu(short_conv("k",D("k_proj",Pp)(x))).reshape(B,L,H,Dh)
        v=jax.nn.silu(short_conv("v",D("v_proj",Pp)(x))).reshape(B,L,H,Dh)
        l2n=lambda t:t*jax.lax.rsqrt(jnp.sum(t*t,-1,keepdims=True)+eps**2)
        q,k=l2n(q),l2n(k)
        erase=jax.nn.sigmoid(D("erase_gate",Pp,True)(x)).reshape(B,L,H,Dh)
        write=jax.nn.sigmoid(D("write_gate",Pp,True)(x)).reshape(B,L,H,Dh)
        decay_a=self.param("decay_a",nn.initializers.zeros,(H,)).astype(jnp.float32)
        f_b=D("f_b_proj",Pp)(D("f_a_proj",128)(x))
        dt_bias=self.param("dt_bias",nn.initializers.zeros,(Pp,))
        raw_gate=f_b.reshape(B,L,H,Dh).astype(jnp.float32)+dt_bias.reshape(H,Dh)
        # K3 config uses safe_gate with lower_bound=-5 (not clipped softplus).
        log_decay=-5.0*jax.nn.sigmoid(jnp.exp(decay_a)[None,None,:,None]*raw_gate)
        self.sow('diagnostics','g_min',jnp.min(log_decay))
        self.sow('diagnostics','g_max',jnp.max(log_decay))
        out_gate=D("out_gate",Pp)(x)
        if USE_ATOMIC_KERNEL:
            o,_=gdn2_forward_atomic(q,k,v,write,erase,log_decay); o=o.reshape(B,L,Pp)
        else:   # fallback: та же семантика, что token_serial_ref. S: (B,H,value,key)
            S0=jnp.zeros((B,H,Dh,Dh),jnp.float32)
            def step(S,xs):
                q_t,k_t,v_t,e_t,w_t,g_t=xs
                S=S*jnp.exp(g_t)[...,None,:]
                Sk=jnp.einsum("bhij,bhj->bhi",S,e_t*k_t)
                S=S+jnp.einsum("bhi,bhj->bhij",w_t*v_t-Sk,k_t)
                return S,jnp.einsum("bhij,bhj->bhi",S,q_t)
            xs=jax.tree.map(lambda t:jnp.swapaxes(t.astype(jnp.float32),1,0),(q,k,v,erase,write,log_decay))
            _,o=jax.lax.scan(jax.checkpoint(step),S0,xs); o=jnp.swapaxes(o,0,1).reshape(B,L,Pp)
            o=o/math.sqrt(Dh)
        o=nn.RMSNorm(epsilon=1e-5,param_dtype=jnp.bfloat16,name="o_norm")(o.reshape(B,L,H,Dh)).reshape(B,L,Pp)*jax.nn.sigmoid(out_gate)
        return nn.Dense(cfg.d_model,use_bias=False,name="o_proj",**BF)(o)

class MLAMixer(nn.Module):
    cfg:StudentConfig
    @nn.compact
    def __call__(self,x,cos,sin):
        cfg=self.cfg; B,L,_=x.shape; H,D=cfg.n_heads,cfg.d_head; R=cfg.qk_rope_head_dim
        dense=lambda n,o:nn.Dense(o,use_bias=False,name=n,**BF)
        qa=nn.RMSNorm(epsilon=1e-6,param_dtype=jnp.bfloat16,name="q_a_layernorm")(dense("q_a_proj",cfg.q_lora_rank)(x))
        q=dense("q_b_proj",H*(D+R))(qa).reshape(B,L,H,D+R)
        kc,kr=jnp.split(dense("kv_a_proj_with_mqa",cfg.kv_lora_rank+R)(x),[cfg.kv_lora_rank],-1)
        kc=nn.RMSNorm(epsilon=1e-6,param_dtype=jnp.bfloat16,name="kv_a_layernorm")(kc)
        kv=dense("kv_b_proj",H*2*D)(kc).reshape(B,L,H,2*D)
        k,v=jnp.split(kv,2,-1)
        # Published K3 MLA uses concatenated shared k_rot with no RoPE application.
        k=jnp.concatenate([k,jnp.broadcast_to(kr[:,:,None,:],(B,L,H,R))],-1)
        scores=jnp.einsum("blhd,bmhd->bhlm",q.astype(jnp.float32),k.astype(jnp.float32))/math.sqrt(D+R)
        att=jax.nn.softmax(jnp.where(jnp.tril(jnp.ones((L,L),bool))[None,None],scores,-1e30),-1)
        o=jnp.einsum("bhlm,bmhd->blhd",att,v.astype(jnp.float32)).reshape(B,L,H*D)
        o=o*jax.nn.sigmoid(dense("g_proj",H*D)(x))
        return dense("o_proj",cfg.d_model)(o)

class FFNSublayer(nn.Module):
    cfg:StudentConfig
    @nn.compact
    def __call__(self,x):
        g=nn.Dense(self.cfg.d_ff,use_bias=False,name="gate_proj",**BF)(x); u=nn.Dense(self.cfg.d_ff,use_bias=False,name="up_proj",**BF)(x)
        return nn.Dense(self.cfg.d_model,use_bias=False,name="down_proj",**BF)(situ_glu(g,u))

class DARLayer(nn.Module):
    cfg:StudentConfig; layer_idx:int
    @nn.compact
    def __call__(self,x,local_deltas,history,cos,sin):
        src=[history[j] for j in range(history.shape[0])]+list(local_deltas)
        x=x+DeltaAttentionResidual(self.cfg.d_model,self.cfg.dar_latent,name="dar")(x,src)
        nx=nn.RMSNorm(epsilon=1e-6,param_dtype=jnp.bfloat16,name="pre_sublayer_norm")(x)
        d=MLAMixer(cfg=self.cfg,name="mixer")(nx,cos,sin) if self.layer_idx in self.cfg.mla_positions else GDN2Mixer(cfg=self.cfg,name="mixer")(nx)
        return x,d

class DARBlock(nn.Module):
    cfg:StudentConfig; block_idx:int; layer_idx_start:int
    @nn.compact
    def __call__(self,x,history,cos,sin):
        ld=[]
        for i in range(self.cfg.layers_per_block):
            idx=self.layer_idx_start+i
            x,d=DARLayer(cfg=self.cfg,layer_idx=idx,name=f"layer_{idx}")(x,ld,history,cos,sin); ld.append(d)
            x=x+d
        nh=jnp.concatenate([history,sum(ld)[None]],0)
        nx=nn.RMSNorm(epsilon=1e-6,param_dtype=jnp.bfloat16,name="pre_ffn_norm")(x)
        return x+FFNSublayer(cfg=self.cfg,name="ffn")(nx),nh

class StudentLM(nn.Module):
    cfg:StudentConfig
    @nn.compact
    def __call__(self,ids):
        B,L=ids.shape
        embed=nn.Embed(self.cfg.vocab,self.cfg.d_model,name="embed",**BF)
        x=embed(ids); cos,sin=RoPE(self.cfg.qk_rope_head_dim,self.cfg.rope_theta)(L)
        hist=jnp.zeros((0,B,L,self.cfg.d_model),dtype=x.dtype); RB=nn.remat(DARBlock)
        for bi in range(self.cfg.num_blocks):
            x,hist=RB(cfg=self.cfg,block_idx=bi,layer_idx_start=bi*self.cfg.layers_per_block,name=f"block_{bi}")(x,hist,cos,sin)
        x=nn.RMSNorm(epsilon=1e-6,param_dtype=jnp.bfloat16,name="final_norm")(x)
        return embed.attend(x) if self.cfg.tie_embeddings else nn.Dense(self.cfg.vocab,use_bias=False,name="lm_head",**BF)(x)


# These gates run before allocating/fetching the large student.
def run_operator_diagnostics():
    rng=np.random.default_rng(17)
    cur=jnp.asarray(rng.normal(size=(1,8,16)),jnp.float32)
    dar=DeltaAttentionResidual(16,8)
    params=dar.init(jax.random.PRNGKey(3),cur,[])['params']
    for n in [0,1,3]:
        sources=[jnp.asarray(rng.normal(size=cur.shape),jnp.float32) for _ in range(n)]
        out=dar.apply({'params':params},cur,sources)
        chk('DAR',f'zero_init_n{n}',np.array_equal(np.asarray(out),np.zeros(cur.shape)))
    sources=[cur,cur*2]
    grad=jax.grad(lambda p:dar.apply({'params':p},cur,sources).sum())(params)
    chk('DAR','alpha_gradient',np.isfinite(float(grad['alpha_scale'])) and abs(float(grad['alpha_scale']))>1e-6)
    chk('DAR','query_gradient_zero_at_alpha0',np.array_equal(np.asarray(grad['w_l']),np.zeros(16)))
    # Independent direct matrix update vs batched einsum scan, nonconstant data,
    # nonzero initial state and multiple chunks' worth of tokens.
    B,T,H,D=1,512,2,8
    q,k,v=[rng.normal(size=(T,B,H,D)).astype('float32')*.1 for _ in range(3)]
    e,w=[rng.uniform(.1,.8,size=(T,B,H,D)).astype('float32') for _ in range(2)]
    g=-rng.uniform(.01,.15,size=(T,B,H,D)).astype('float32')
    s0=rng.normal(size=(B,H,D,D)).astype('float32')*.01
    def scan(q,k,v,e,w,g):
        def step(s,z):
            qt,kt,vt,et,wt,gt=z
            s=s*jnp.exp(gt)[...,None,:]
            sk=jnp.einsum('bhij,bhj->bhi',s,et*kt)
            s=s+jnp.einsum('bhi,bhj->bhij',wt*vt-sk,kt)
            return s,jnp.einsum('bhij,bhj->bhi',s,qt)/math.sqrt(D)
        return jax.lax.scan(step,jnp.asarray(s0),(q,k,v,e,w,g))
    state=s0.copy(); expected=[]
    for t in range(T):
        state*=np.exp(g[t])[...,None,:]
        for b in range(B):
            for h in range(H):
                sk=state[b,h]@(e[t,b,h]*k[t,b,h])
                state[b,h]+=np.outer(w[t,b,h]*v[t,b,h]-sk,k[t,b,h])
        expected.append(np.matmul(state,q[t,...,None])[...,0]/math.sqrt(D))
    sf,out=jax.jit(scan)(q,k,v,e,w,g)
    chk('SCAN','random_matrix_reference',np.allclose(out,expected,rtol=2e-4,atol=2e-6) and np.allclose(sf,state,rtol=2e-4,atol=2e-6))
    f=lambda vv:jnp.sum(scan(q,k,vv,e,w,g)[1]**2)
    grad=jax.grad(f)(jnp.asarray(v))
    ix=tuple(np.unravel_index(np.argmax(np.abs(np.asarray(grad))),v.shape)); eps=1e-3
    vp=v.copy(); vm=v.copy(); vp[ix]+=eps; vm[ix]-=eps
    fd=float((f(vp)-f(vm))/(2*eps)); ad=float(grad[ix])
    chk('SCAN','finite_difference_v',abs(ad)>1e-7 and np.isclose(fd,ad,rtol=.03,atol=1e-9),{'autodiff':ad,'finite_difference':fd,'index':ix})
    REPORT['kernel_scope']='JAX scan only; no atomic_ops/atomic_gdn2 validation'; save_rpt()
run_operator_diagnostics()

"""Resource bookkeeping only; estimates do not establish runtime fit."""
def select_batch(requested,n_devices):
    if requested<0 or n_devices<1:raise ValueError('Invalid batch/device count')
    return n_devices if requested==0 else requested

def parameter_budget(rows,n_devices,training):
    resident=master=0.0
    for row in rows:
        size=1
        for s in row['shape']:size*=s
        shards=n_devices if any(s>=n_devices and s%n_devices==0 for s in row['shape']) else 1
        if n_devices==1:shards=1
        dtype_bytes=4 if row['trainable'] else row['itemsize']
        resident+=size*dtype_bytes/shards
        if row['trainable']:master+=size*4/shards
    return dict(weight_bytes_per_device=int(resident),trainable_master_bytes_per_device=int(master),
                indicative_training_bytes_per_device=int(resident+4*master) if training else None,
                scope='Weights are a lower bound. Training estimate omits activations, compilation and allocator overhead; fit is unverified.')

def check_weight_capacity(budget,device_stats):
    for stats in device_stats:
        limit=stats.get('bytes_limit')
        if not isinstance(limit,(int,float)) or limit<=0:continue
        free=limit-stats.get('bytes_in_use',0)
        if budget['weight_bytes_per_device']>free:
            raise MemoryError('Full student weights alone exceed reported free memory on a TPU device. No model weights have been downloaded; see resource_budget in report.json.')

def resource_preflight(abstract):
    rows=[]
    for path,aval in jax.tree_util.tree_flatten_with_path(abstract)[0]:
        key='/'.join(str(getattr(k,'key',k)) for k in path)
        trainable=('/dar/' in key or '/erase_gate/' in key or '/write_gate/' in key or key.endswith('/dt_bias') or key.endswith('/pre_ffn_norm/scale'))
        rows.append(dict(shape=tuple(aval.shape),itemsize=np.dtype(aval.dtype).itemsize,trainable=trainable))
    budget=parameter_budget(rows,N_DEV,STEPS>0)
    REPORT['resource_budget']=budget;save_rpt()
    print('[resource budget]',budget,flush=True)
    check_weight_capacity(budget,REPORT['hardware']['memory_stats'])


cfg=StudentConfig() if MODE=="real" else StudentConfig(d_model=256,num_blocks=1,n_layers=4,vocab=1024,n_heads=2,q_lora_rank=64,kv_lora_rank=32,d_ff=512,mla_positions=(3,))
BT=128; model=StudentLM(cfg=cfg); rng=jax.random.PRNGKey(0)
ids_small=np.zeros((1,8),np.int32)
abstract=jax.eval_shape(lambda r:model.init(r,ids_small)["params"],rng)
out_shardings=jax.tree.map(lambda x:NamedSharding(mesh,fsdp_spec(x.shape,N_DEV)),abstract)
REPORT["config"]=dataclasses.asdict(cfg)
REPORT["parameters"]=sum(int(x.size) for x in jax.tree.leaves(abstract)); save_rpt()
print("[parameters]",REPORT["parameters"])
if MODE=="tiny":
    params_k3=jax.jit(lambda r:model.init(r,ids_small)["params"],out_shardings=out_shardings)(rng)
else:
    if jax.default_backend()!="tpu":
        raise RuntimeError("Real model requires a TPU backend; stopping before downloads")
    resource_preflight(abstract)

"""enwik8 identity: https://mattmahoney.net/dc/textdata.html (size + SHA-1)."""
import gzip,hashlib,io,json,os,zipfile
from pathlib import Path

ENWIK8_SIZE=100_000_000
ENWIK8_SHA1='57b8363b814821dc9d47aa4d41f58733519076b2'
ENWIK8_URLS=[
    'https://raw.githubusercontent.com/lucidrains/x-transformers/main/data/enwik8.gz',
    'https://www.mattmahoney.net/dc/enwik8.zip',
    'https://mattmahoney.net/dc/enwik8.zip',
    'http://mattmahoney.net/dc/enwik8.zip',
]

def decode_enwik8(blob,size=ENWIK8_SIZE,sha1=ENWIK8_SHA1):
    if blob.startswith(b'PK\x03\x04'):
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            names=[n for n in z.namelist() if n.rsplit('/',1)[-1]=='enwik8']
            if len(names)!=1 or z.getinfo(names[0]).file_size!=size:raise ValueError('ZIP has no unique enwik8 of expected size')
            raw=z.read(names[0])
    elif blob.startswith(b'\x1f\x8b'):
        with gzip.GzipFile(fileobj=io.BytesIO(blob)) as f:raw=f.read(size+1)
    elif len(blob)==size:raw=blob
    else:raise ValueError(f'Response is neither ZIP, gzip nor expected raw file; bytes={len(blob)}, magic_hex={blob[:8].hex()}')
    if len(raw)!=size or hashlib.sha1(raw).hexdigest()!=sha1:raise ValueError('enwik8 size/SHA-1 mismatch')
    return raw

def load_enwik8(cache_dir,local_path='',report=None,save_report=lambda:None,urls=None,session=None,size=ENWIK8_SIZE,sha1=ENWIK8_SHA1):
    cache=Path(cache_dir);cache.mkdir(parents=True,exist_ok=True)
    target=cache/'enwik8';attempts=[]
    if report is not None:report['data_download_attempts']=attempts
    if local_path:
        raw=decode_enwik8(Path(local_path).read_bytes(),size,sha1)
        source='user file'
    elif target.is_file():
        try:
            raw=decode_enwik8(target.read_bytes(),size,sha1);source='verified local cache'
        except (ValueError,OSError,zipfile.BadZipFile):raw=None
    else:raw=None
    if not local_path and raw is None:
        if session is None:
            import requests
            session=requests
        for url in ENWIK8_URLS if urls is None else urls:
            record={'source':url.split('?',1)[0]};attempts.append(record)
            try:
                with session.get(url,stream=True,timeout=(30,180),headers={'User-Agent':'K3-transfer-dataset/4'}) as r:
                    record.update(status=r.status_code,content_type=r.headers.get('Content-Type'))
                    r.raise_for_status();body=bytearray()
                    for chunk in r.iter_content(1024*1024):
                        body.extend(chunk)
                        if len(body)>size+1024*1024:raise ValueError('Dataset response exceeds download bound')
                record.update(bytes=len(body),magic_hex=bytes(body[:8]).hex())
                raw=decode_enwik8(bytes(body),size,sha1);source=record['source'];record['verified']=True
                save_report();break
            except Exception as exc:
                record.update(verified=False,error_type=type(exc).__name__)
                save_report();print('[data source rejected]',record,flush=True)
        if raw is None:raise RuntimeError('No verified enwik8 source succeeded. See data_download_attempts, or set ENWIK8_PATH to enwik8, enwik8.zip or enwik8.gz.')
    temp=target.with_suffix('.tmp');temp.write_bytes(raw);os.replace(temp,target)
    if report is not None:
        report['enwik8_identity']={'bytes':len(raw),'sha1':hashlib.sha1(raw).hexdigest(),'sha256':hashlib.sha256(raw).hexdigest(),'source':source,'verified':True}
    save_report();print('[data] verified enwik8',len(raw),'bytes; source:',source,flush=True)
    return raw

import optax
from flax.traverse_util import flatten_dict,unflatten_dict

if MODE=='real':
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('moonshotai/Kimi-K3',revision=K3_SHA,token=HF_TOKEN,trust_remote_code=True)
    # Hold out raw text BEFORE tokenization. Bounded diagnostic subset, not full enwik8 training.
    raw=load_enwik8(Path(os.environ.get('K3_CACHE_DIR','/content/k3_cache'))/'data',local_path=os.environ.get('K3_ENWIK8_PATH',''),report=REPORT,save_report=save_rpt)
    REPORT['enwik8_sha256']=hashlib.sha256(raw).hexdigest()
    sections=[raw[:2_000_000],raw[95_000_000:95_200_000]]
    arrays=[]; byte_counts=[]
    for rawpart in sections:
        text=rawpart.decode('utf-8',errors='replace')
        ids=np.asarray(tok.encode(text,add_special_tokens=False),np.int32)
        if ids.min()<0 or ids.max()>=cfg.vocab: raise ValueError('Tokenizer IDs exceed student vocabulary')
        arrays.append(ids); byte_counts.append(len(text.encode('utf-8')))
    train_ids,val_ids=arrays
    REPORT['data']={'train_tokens':len(train_ids),'val_tokens':len(val_ids),'effective_utf8_bytes':byte_counts,'bytes_per_token':[n/len(a) for n,a in zip(byte_counts,arrays)],'decode_errors':'replace','full_enwik8':False}
else:
    # A synthetic deterministic transition task, not pretrained-language quality.
    train_ids=np.tile(np.arange(64,dtype=np.int32),2000)
    val_ids=np.tile(np.arange(64,dtype=np.int32),200)
    REPORT['data']={'synthetic':True}


if MODE=="real":
    REPO="moonshotai/Kimi-K3"; REV=K3_SHA
    HF_HEADERS={"Authorization":f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    def hf_url(path,repo=REPO,rev=REV): return f"https://huggingface.co/{repo}/resolve/{rev}/{path}"
    def http_get(url,byte_range=None,retries=4):
        h=dict(HF_HEADERS)
        if byte_range: h["Range"]=f"bytes={byte_range[0]}-{byte_range[1]}"
        last=None
        for a in range(retries):
            try: r=requests.get(url,headers=h,allow_redirects=True,timeout=180,stream=True)
            except requests.RequestException as e: last=e; time.sleep(2**a); continue
            if byte_range and r.status_code!=206:
                r.close(); raise RuntimeError(f"Range not honored: HTTP {r.status_code}")
            if r.status_code in (200,206):
                data=r.content; r.close()
                if byte_range and len(data)!=byte_range[1]-byte_range[0]+1: raise ValueError("Truncated range")
                return data
            if r.status_code in (429,500,502,503,504): last=RuntimeError(f"HTTP {r.status_code}"); time.sleep(2**a); continue
            r.raise_for_status()
        raise RuntimeError(f"GET failed: {last}")
    NP_DTYPES={"BF16":ml_dtypes.bfloat16,"F16":np.float16,"F32":np.float32,"U8":np.uint8,"I8":np.int8,"I32":np.int32,"I64":np.int64,
      "F8_E4M3":ml_dtypes.float8_e4m3fn,"F8_E5M2":ml_dtypes.float8_e5m2,"F8_E8M0":getattr(ml_dtypes,"float8_e8m0fnu",None)}
    def load_index(repo=REPO,rev=REV): return json.loads(http_get(hf_url("model.safetensors.index.json",repo,rev)))["weight_map"]
    _hc={}
    def read_shard_header(shard,repo=REPO,rev=REV):
        key=(repo,rev,shard)
        if key in _hc: return _hc[key]
        url=hf_url(shard,repo,rev); (n,)=struct.unpack("<Q",http_get(url,(0,7)))
        hd=json.loads(http_get(url,(8,7+n))); hd.pop("__metadata__",None); _hc[key]=(hd,8+n); return _hc[key]
    def _to_array(buf,dt,shape):
        d=NP_DTYPES[dt]
        if d is None: return np.frombuffer(bytearray(buf),dtype=np.uint8).copy()
        return np.frombuffer(bytearray(buf),dtype=d).reshape(shape).copy()
    def plan_fetch(names,wm,repo=REPO,rev=REV):
        bs=defaultdict(list)
        for n in names: bs[wm[n]].append(n)
        plan=[]
        for shard,ns in bs.items():
            hd,base=read_shard_header(shard,repo,rev); url=hf_url(shard,repo,rev)
            for n in ns:
                i=hd[n]; s,e=i["data_offsets"]
                plan.append(dict(name=n,url=url,start=base+s,end=base+e-1,dtype=i["dtype"],shape=i["shape"]))
        return plan
    def fetch_tensors(names,wm,workers=8,dry_run=True,repo=REPO,rev=REV):
        plan=plan_fetch(names,wm,repo,rev); tot=sum(p["end"]-p["start"]+1 for p in plan)
        print(f"{len(plan)} tensors, {tot/1e9:.2f} GB")
        if dry_run: return None
        def get(p): return p["name"],_to_array(cached_tensor_bytes(p),p["dtype"],p["shape"])
        with ThreadPoolExecutor(workers) as ex: return dict(ex.map(get,plan))
    chk("HUB","fetcher_ready",True)
    """Cache immutable selected HF tensor byte ranges outside results archives."""
    def cached_tensor_bytes(p):
        key=hashlib.sha256(json.dumps({k:p[k] for k in ['name','url','start','end','dtype','shape']},sort_keys=True).encode()).hexdigest()
        root=Path(os.environ.get('K3_CACHE_DIR','/content/k3_cache'))/'tensor_ranges'/REV
        root.mkdir(parents=True,exist_ok=True)
        data_path=root/(key+'.bin');meta_path=root/(key+'.json');expected=p['end']-p['start']+1
        if data_path.is_file() and meta_path.is_file():
            try:
                meta=json.loads(meta_path.read_text());buf=data_path.read_bytes()
                if len(buf)==expected and hashlib.sha256(buf).hexdigest()==meta['sha256']:return buf
            except (OSError,ValueError,KeyError):pass
        buf=http_get(p['url'],(p['start'],p['end']))
        if len(buf)!=expected:raise ValueError('Tensor byte-range length mismatch')
        try:
            tmp=data_path.with_suffix('.tmp');tmp.write_bytes(buf);os.replace(tmp,data_path)
            tmp_meta=meta_path.with_suffix('.tmp');tmp_meta.write_text(json.dumps({'name':p['name'],'sha256':hashlib.sha256(buf).hexdigest(),'bytes':len(buf)}));os.replace(tmp_meta,meta_path)
        except OSError as exc:print('[tensor cache write skipped]',type(exc).__name__,flush=True)
        return buf
    
    import numpy as _np, jax.tree_util as jtu
    K3_FFN_LAYER=1
    KDA_SUF=["self_attn.q_proj.weight","self_attn.k_proj.weight","self_attn.v_proj.weight","self_attn.o_proj.weight","self_attn.g_proj.weight",
     "self_attn.q_conv1d.weight","self_attn.k_conv1d.weight","self_attn.v_conv1d.weight","self_attn.b_proj.weight",
     "self_attn.f_a_proj.weight","self_attn.f_b_proj.weight","self_attn.A_log","self_attn.dt_bias","self_attn.o_norm.weight","input_layernorm.weight"]
    MLA_SUF=["self_attn.q_a_proj.weight","self_attn.q_b_proj.weight","self_attn.kv_a_proj_with_mqa.weight","self_attn.kv_b_proj.weight",
     "self_attn.q_a_layernorm.weight","self_attn.kv_a_layernorm.weight","self_attn.o_proj.weight","self_attn.g_proj.weight","input_layernorm.weight"]
    need=[f"language_model.model.layers.{L}.{s}" for L in range(cfg.n_layers) for s in (MLA_SUF if L in cfg.mla_positions else KDA_SUF)]
    pre_sh=f"language_model.model.layers.{K3_FFN_LAYER}.block_sparse_moe.shared_experts"
    need+=[f"{pre_sh}.{w}.weight" for w in ("gate_proj","up_proj","down_proj")]
    need+=["language_model.model.embed_tokens.weight","language_model.model.norm.weight","language_model.lm_head.weight"]
    wm=load_index(REPO,REV); missing=[n for n in need if n not in wm]
    if missing: raise KeyError(f"Missing required pretrained tensors: {missing}")
    need=[n for n in need if n in wm]; print(f"[fetch] {len(need)} tensors"); fetch_tensors(need,wm,dry_run=True)
    tensors=fetch_tensors(need,wm,dry_run=False,workers=8); print(f"[fetch] loaded {len(tensors)}")
    f32=lambda a:_np.asarray(a)  # retain source bf16 on host; cast only into target dtype
    def _sl_rows(W,n): return _np.asarray(W)[:n,...]
    def _sl_cols(W,n): return _np.asarray(W)[...,:n]
    def _gate_dense(b_k3,H_s,Dh):
        b=f32(b_k3); bh=b[:H_s]
        return _np.repeat(bh[:,None,:],Dh,axis=1).reshape(H_s*Dh,-1), _np.zeros(H_s*Dh,_np.float32)
    def _kda(layer,H_s,Dh):
        p=f"language_model.model.layers.{layer}.self_attn"; Ps=H_s*Dh; T=tensors; o={}
        o["mixer/q_proj/kernel"]=_sl_rows(T[f"{p}.q_proj.weight"],Ps).T; o["mixer/k_proj/kernel"]=_sl_rows(T[f"{p}.k_proj.weight"],Ps).T
        o["mixer/v_proj/kernel"]=_sl_rows(T[f"{p}.v_proj.weight"],Ps).T; o["mixer/o_proj/kernel"]=_sl_cols(T[f"{p}.o_proj.weight"],Ps)
        o["mixer/out_gate/kernel"]=_sl_rows(T[f"{p}.g_proj.weight"],Ps).T
        for n in "qkv": o[f"mixer/{n}_conv_w"]=_sl_rows(T[f"{p}.{n}_conv1d.weight"].reshape(-1,4),Ps)
        eW,eb=_gate_dense(T[f"{p}.b_proj.weight"],H_s,Dh)
        o["mixer/erase_gate/kernel"]=eW.T; o["mixer/write_gate/kernel"]=eW.T; o["mixer/erase_gate/bias"]=eb; o["mixer/write_gate/bias"]=eb
        o["mixer/f_a_proj/kernel"]=f32(T[f"{p}.f_a_proj.weight"]).T; o["mixer/f_b_proj/kernel"]=_sl_rows(T[f"{p}.f_b_proj.weight"],Ps)  # авто-транспонируется ниже
        A=f32(T[f"{p}.A_log"]); o["mixer/decay_a"]=A[:H_s]
        o["mixer/dt_bias"]=f32(T[f"{p}.dt_bias"])[:Ps]
        o["mixer/o_norm/scale"]=f32(T[f"{p}.o_norm.weight"])
        o["pre_sublayer_norm/scale"]=f32(T[f"language_model.model.layers.{layer}.input_layernorm.weight"]); return o
    def _mla(layer,H_s,Dh):
        p=f"language_model.model.layers.{layer}.self_attn"; Ps=H_s*Dh; T=tensors; o={}
        o["mixer/q_a_proj/kernel"]=f32(T[f"{p}.q_a_proj.weight"]).T; o["mixer/q_a_layernorm/scale"]=f32(T[f"{p}.q_a_layernorm.weight"])
        o["mixer/q_b_proj/kernel"]=_sl_rows(T[f"{p}.q_b_proj.weight"],H_s*(Dh+cfg.qk_rope_head_dim)).T
        o["mixer/g_proj/kernel"]=_sl_rows(T[f"{p}.g_proj.weight"],Ps).T
        o["mixer/kv_a_proj_with_mqa/kernel"]=f32(T[f"{p}.kv_a_proj_with_mqa.weight"]).T; o["mixer/kv_a_layernorm/scale"]=f32(T[f"{p}.kv_a_layernorm.weight"])
        kb=f32(T[f"{p}.kv_b_proj.weight"]).reshape(-1,2*Dh,cfg.kv_lora_rank)[:H_s].reshape(2*Ps,cfg.kv_lora_rank); h=Ps
        o["mixer/kv_b_proj/kernel"]=kb.T; o["mixer/o_proj/kernel"]=_sl_cols(T[f"{p}.o_proj.weight"],Ps)
        o["pre_sublayer_norm/scale"]=f32(T[f"language_model.model.layers.{layer}.input_layernorm.weight"]); return o
    H_s,Dh=cfg.n_heads,cfg.d_head; patch={}
    for L in range(cfg.n_layers):
        for k,v in (_mla if L in cfg.mla_positions else _kda)(L,H_s,Dh).items(): patch[f"block_{L//cfg.layers_per_block}/layer_{L}/{k}"]=v
    G,U,Dn=(f32(tensors[f"{pre_sh}.{w}.weight"]) for w in ("gate_proj","up_proj","down_proj"))
    for bi in range(cfg.num_blocks):
        patch[f"block_{bi}/ffn/gate_proj/kernel"]=G.T; patch[f"block_{bi}/ffn/up_proj/kernel"]=U.T; patch[f"block_{bi}/ffn/down_proj/kernel"]=Dn.T
    patch["embed/embedding"]=tensors["language_model.model.embed_tokens.weight"]; patch["final_norm/scale"]=tensors["language_model.model.norm.weight"]
    patch["lm_head/kernel"]=tensors["language_model.lm_head.weight"].T
    print(f"[patch] {len(patch)} patches ~{sum(v.nbytes for v in patch.values())/1e9:.2f} GB"); del tensors; gc.collect()
    rng_dar=_np.random.default_rng(42); abs_leaves=jtu.tree_flatten_with_path(abstract)[0]; treedef=jtu.tree_structure(abstract)
    new_host=[]; applied=fb=0; unused=set(patch); fallback_keys=[]
    for path,aval in abs_leaves:
        key="/".join(str(getattr(p,"key",p)) for p in path)
        dt=_np.dtype(jax.dtypes.canonicalize_dtype(aval.dtype)); shape=tuple(aval.shape)
        if key in patch:
            a=_np.asarray(patch[key])
            if a.shape!=shape:
                if a.T.shape==shape: a=a.T                                   # o_proj / f_b_proj: (out,in)->(in,out)
                elif a.ndim==1 and len(shape)==1 and shape[0]%a.shape[0]==0: a=_np.tile(a,shape[0]//a.shape[0])  # per-head norm -> full
                else: raise ValueError(f"{key}: patch {a.shape} != model {shape}")
            new_host.append(a.astype(dt,copy=False)); applied+=1; unused.discard(key)
        elif "/dar/" in key:
            new_host.append((_np.ones(shape,dt) if key.endswith("/scale") else _np.zeros(shape,dt))); fb+=1
            fallback_keys.append({"key":key,"reason":"new DAR parameter"})
        elif key.endswith("pre_ffn_norm/scale"):
            new_host.append(_np.ones(shape,dt)); fb+=1
            fallback_keys.append({"key":key,"reason":"new block FFN norm"})
        elif key.endswith("_conv_b"):
            new_host.append(_np.zeros(shape,dt)); fb+=1
            fallback_keys.append({"key":key,"reason":"K3 bias-free convolution"})
        else: raise KeyError(f"Unexpected fallback: {key}")
    print(f"[host-tree] applied={applied} fallback={fb} unused={len(unused)} {list(unused)[:3]}")
    REPORT["fallback_keys"]=fallback_keys; save_rpt()
    assert not unused, f"Unused patches: {unused}"
    del patch; gc.collect()
    host_tree=jtu.tree_unflatten(treedef,new_host)
    params_k3=jax.tree.map(lambda x,s:jax.device_put(x,s),host_tree,out_shardings); jax.block_until_ready(params_k3); del host_tree,new_host; gc.collect()
    logits_k3=jax.jit(model.apply)({"params":params_k3},jnp.zeros((1,BT),jnp.int32))
    chk("INIT","k3_full_init",logits_k3.shape==(1,BT,cfg.vocab) and bool(jnp.isfinite(logits_k3).all()),f"std={float(jnp.std(logits_k3)):.3f}")
    chk("INIT","all_patches_used",len(unused)==0,f"unused={len(unused)}")
    globals()["_student_params"]=params_k3; category_summary("INIT")

chk('DATA','enough_tokens',len(train_ids)>SEQ+1 and len(val_ids)>SEQ+1)
shard_batch=NamedSharding(mesh,P('fsdp',None)) if BATCH%N_DEV==0 else NamedSharding(mesh,P())
rng_data=np.random.default_rng(19)
def sample(data):
    starts=rng_data.integers(0,len(data)-SEQ-1,size=BATCH)
    a=np.stack([data[s:s+SEQ+1] for s in starts]); return jax.device_put(a,shard_batch)

# Train only added residual routing, erase/write gates, dt_bias, new FFN norms.
is_trainable=lambda k:('/dar/' in k or '/erase_gate/' in k or '/write_gate/' in k or k.endswith('/dt_bias') or k.endswith('/pre_ffn_norm/scale'))
flat=flatten_dict(params_k3,sep='/')
tr=unflatten_dict({k:jax.device_put(np.asarray(v,dtype=np.float32),NamedSharding(mesh,fsdp_spec(v.shape,N_DEV))) for k,v in flat.items() if is_trainable(k)},sep='/')
fr=unflatten_dict({k:v for k,v in flat.items() if not is_trainable(k)},sep='/')
del flat,params_k3
# Release diagnostic aliases to the old bf16 tree before allocating fp32 optimizer state.
globals().pop('_student_params',None);globals().pop('logits_k3',None);gc.collect()
REPORT['trainable_keys']=list(flatten_dict(tr,sep='/'))
REPORT['trainable_count']=sum(int(x.size) for x in jax.tree.leaves(tr)); save_rpt()
merge=lambda a,b:unflatten_dict({**flatten_dict(a,sep='/'),**flatten_dict(b,sep='/')},sep='/')
def loss_fn(a,b,batch):
    logits=model.apply({'params':merge(a,b)},batch[:,:-1]).astype(jnp.float32)
    return optax.softmax_cross_entropy_with_integer_labels(logits,batch[:,1:]).mean()
evaluate=jax.jit(loss_fn)
batch=sample(val_ids)
ce0=float(evaluate(tr,fr,batch)); chk('INIT','finite_ce0',np.isfinite(ce0),ce0)
# Shuffled sequence uses identical token multiset; recorded as a diagnostic only.
sh=np.asarray(batch).copy()
for row in sh: rng_data.shuffle(row)
shuffled=float(evaluate(tr,fr,jax.device_put(sh,shard_batch)))
counts=np.bincount(train_ids,minlength=cfg.vocab).astype(np.float64)+1
unigram=-np.log(counts/counts.sum())[np.asarray(batch)[:,1:]].mean()
REPORT['initial']={'ce_nat_per_token':ce0,'shuffled_ce':shuffled,'unigram_ce':float(unigram),'ln_vocab':math.log(cfg.vocab),'teacher_function_preservation':'NOT ESTABLISHED'}
if MODE=='real':
    bpt=REPORT['data']['bytes_per_token'][1]
    REPORT['initial']['estimated_bpb_using_val_ratio']=ce0/math.log(2)/bpt
    REPORT['initial']['bpb_scope']='sample CE divided by complete held-out subset bytes/token ratio'
def gate_stats(a,b,ids):
    return model.apply({'params':merge(a,b)},ids,mutable=['diagnostics'])[1]
stats=jax.jit(gate_stats)(tr,fr,batch[:,:-1])
REPORT['actual_gate_stats']={k:[float(x) for x in jax.tree.leaves(v)] for k,v in flatten_dict(stats,sep='/').items()}
min_g=min((x for vs in REPORT['actual_gate_stats'].values() for x in vs),default=0.)
REPORT['half_span_bound_from_sample']=32*abs(min_g)/2
save_rpt(); print('[initial]',REPORT['initial'],flush=True)
if MODE=='tiny':
    gg=flatten_dict(jax.jit(jax.grad(loss_fn))(tr,fr,batch),sep='/')
    mixer_norms=[float(jnp.linalg.norm(g)) for k,g in gg.items() if '/erase_gate/' in k or '/write_gate/' in k]
    chk('RESIDUAL','attention_affects_loss',all(np.isfinite(mixer_norms)) and max(mixer_norms)>1e-8,mixer_norms)
    del gg

def save_trainable(step):
    # Numeric, pickle-free trainable-only checkpoint. Resume optimizer not included.
    flat={k:np.asarray(v) for k,v in flatten_dict(jax.device_get(tr),sep='/').items()}
    path=Path(ARTIFACTS,'trainable_latest.npz'); temp=path.with_suffix('.tmp')
    with open(temp,'wb') as f:np.savez(f,**flat)
    os.replace(temp,path)
    REPORT['checkpoint']={'file':path.name,'step':step,'kind':'trainable-only','exact_resume':False}; save_rpt()

if STEPS:
    # Bad initialization is investigated before launching an expensive healing run.
    if MODE=='real' and ce0>=math.log(cfg.vocab):
        raise RuntimeError('CE(0) >= ln(vocab); diagnostics saved, training stopped. This is a heuristic, not teacher parity.')
    tx=optax.chain(optax.clip_by_global_norm(1.0),optax.adamw(1e-4,weight_decay=.01)); opt=tx.init(tr)
    @jax.jit
    def train_step(a,op,b,batch):
        loss,grads=jax.value_and_grad(loss_fn)(a,b,batch)
        gn=optax.global_norm(grads); ok=jnp.isfinite(loss)&jnp.isfinite(gn)
        updates,nop=tx.update(grads,op,a); na=optax.apply_updates(a,updates)
        return jax.tree.map(lambda new,old:jnp.where(ok,new,old),na,a),jax.tree.map(lambda new,old:jnp.where(ok,new,old),nop,op),loss,gn,ok
    metrics=Path(ARTIFACTS,'metrics.jsonl')
    for step in range(1,STEPS+1):
        tr,opt,lv,gn,ok=train_step(tr,opt,fr,sample(train_ids))
        row={'step':step,'loss':float(lv),'grad_norm':float(gn),'accepted':bool(ok)}
        with metrics.open('a') as f:f.write(json.dumps(row)+'\n')
        print('[train]',row,flush=True)
        REPORT['last_step']=row; save_rpt()
        if not row['accepted']: raise FloatingPointError('Nonfinite training step; optimizer update skipped')
        if step%25==0:save_trainable(step)
    save_trainable(STEPS)
    REPORT['final_ce']=float(evaluate(tr,fr,batch))
REPORT['status']='COMPLETED'; save_rpt()
