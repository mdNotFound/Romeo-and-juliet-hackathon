import copy, hashlib, itertools, json, math, random, uuid
from pathlib import Path

VERSION = '1.0.0'
GENDERS = ['woman', 'man', 'non_binary']
HARD = ['age_min','age_max','who_to_meet','relationship_structure','smoking',
        'partner_smoking','has_children','partner_children','wants_children',
        'acceptable_zones','schedule']
SOFT = ['relationship_goal','relationship_pace','lifestyle','conversations',
        'emotional_availability','space_for_relationship','relocate']
OPTIONS = {
    'relationship_goal':['long_term','exploring'],
    'relationship_pace':['slow','steady','quick'],
    'lifestyle':['quiet','mixed','social'],
    'conversations':['ideas','stories','practical','playful'],
    'emotional_availability':['ready','taking_time'],
    'space_for_relationship':['limited','moderate','ample'],
    'relocate':['yes','no','unsure'],
}

def rand_for(*parts):
    return random.Random(int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:16],'big'))

def dump_json(path, obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

def dump_lines(path, rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows))

def generate(seed=20260926,n=200,pool_id='dev_01',variant='development'):
    r=random.Random(seed)
    world={'schema_version':VERSION,'synthetic':True,'seed':seed,'variant':variant,
            'pool_id':pool_id,'members':[]}
    for _ in range(n):
        ident='syn_'+uuid.UUID(int=r.getrandbits(128)).hex
        age=r.randint(21,46); gender=r.choices(GENDERS,[.46,.46,.08])[0]
        zone=r.choices(['zone_a','zone_b','zone_c','zone_d'],[.60,.23,.12,.05])[0]
        if variant=='sparse': zone=r.choice(['zone_'+str(i) for i in range(12)])
        wants=r.sample(GENDERS,r.choices([1,2,3],[.7,.2,.1])[0])
        truth={'age_min':max(18,age-r.randint(3,10)), 'age_max':min(65,age+r.randint(3,10)),
            'who_to_meet':wants,'relationship_structure':r.choices(['monogamous','non_monogamous'],[.9,.1])[0],
            'smoking':r.choices(['no','occasionally','yes'],[.75,.2,.05])[0],
            'partner_smoking':r.choice(['no_smoking','any']),
            'has_children':r.random()<.15,'partner_children':r.choice(['any','no_children']),
            'wants_children':r.choice(['yes','no','unsure']),
            'acceptable_zones':[zone],
            'schedule':r.sample(['weekday_evening','weekend_day','weekend_evening'],r.randint(1,3))}
        if variant!='sparse' and r.random()<.3: truth['acceptable_zones']=list(dict.fromkeys([zone,r.choice(['zone_a','zone_b','zone_c','zone_d'])]))
        truth.update({k:r.choice(v) for k,v in OPTIONS.items()})
        complete=r.random()<( .2 if variant=='cold_start' else .35)
        obs={k:copy.deepcopy(v) if complete or r.random()<.10 else None for k,v in truth.items()}
        missing={k:( 'observed' if obs[k] is not None else ('declined' if r.random()<.07 else 'not_asked')) for k in truth}
        arrival=0 if r.random()<.7 else r.randint(1,20)
        member={'member_id':ident,'pool_id':pool_id,'synthetic':True,'age':age,'gender':gender,'zone':zone,
                'arrived_day':arrival,'fields':obs,'field_status':missing,
                'field_observed_day':{k:arrival if v is not None else None for k,v in obs.items()},
                'source':'synthetic_questionnaire','truth':truth,
                'exit_day':r.randint(22,60) if r.random()<.12 else 10000,
                'bias':r.gauss(0,.7),'response_rate':r.uniform(.55,.98),
                'second_bias':r.gauss(0,.5)}
        world['members'].append(member)
    return world



def eligibility(a,b):
    """Public pair-level hard checks only; availability is separately checked by simulator."""
    if a['member_id']==b['member_id']: return {'status':'infeasible','reasons':['same_member']}
    if a['pool_id']!=b['pool_id']: return {'status':'infeasible','reasons':['different_pool']}
    if min(a['age'],b['age'])<18: return {'status':'infeasible','reasons':['underage']}
    fa,fb=a['fields'],b['fields']; missing=[]; failures=[]
    for member,f in ((a,fa),(b,fb)):
        missing += [member['member_id']+':'+k for k in HARD if f.get(k) is None]
    for x,y,fx,fy in ((a,b,fa,fb),(b,a,fb,fa)):
        if fx.get('who_to_meet') is not None and y['gender'] not in fx['who_to_meet']: failures.append('who_to_meet')
        if fx.get('age_min') is not None and y['age']<fx['age_min']: failures.append('age')
        if fx.get('age_max') is not None and y['age']>fx['age_max']: failures.append('age')
        if fx.get('acceptable_zones') is not None and y['zone'] not in fx['acceptable_zones']: failures.append('geography')
        if fx.get('partner_smoking')=='no_smoking' and fy.get('smoking') in ('yes','occasionally'): failures.append('smoking')
        if fx.get('partner_children')=='no_children' and fy.get('has_children') is True: failures.append('children')
    if fa.get('relationship_structure') is not None and fb.get('relationship_structure') is not None and fa['relationship_structure']!=fb['relationship_structure']: failures.append('relationship_structure')
    if {fa.get('wants_children'),fb.get('wants_children')}=={'yes','no'}: failures.append('children_plans')
    if fa.get('schedule') is not None and fb.get('schedule') is not None and not set(fa['schedule'])&set(fb['schedule']): failures.append('schedule')
    return {'status':'infeasible' if failures else ('needs_clarification' if missing else 'feasible'),
            'reasons':sorted(set(failures)),'missing':missing}


class Simulator:
    """Trusted local development simulator, NOT a sandbox for untrusted submissions."""
    def __init__(self, world):
        self.world=copy.deepcopy(world); self.day=0
        self.members={m['member_id']:m for m in self.world['members']}
        self.busy={}; self.past=set(); self.retired=set(); self.introductions=[]; self.events=[]
        self.ask_spent=0; self.asks_total=0; self.ask_log=[]
    def available(self,m):
        i=m['member_id']
        paused=any(e['event']=='pause_after_mutual_interest' and e['observed_day']<=self.day and i in next(( (x['user_a'],x['user_b']) for x in self.introductions if x['introduction_id']==e['introduction_id']),()) for e in self.events)
        return m['arrived_day']<=self.day<m['exit_day'] and not paused and self.busy.get(i,-1)<=self.day
    def observe(self):
        visible=[]
        for m in self.members.values():
            if m['arrived_day']>self.day: continue
            o={k:copy.deepcopy(m[k]) for k in ['member_id','pool_id','synthetic','age','gender','zone','arrived_day','fields','field_status','field_observed_day','source']}
            o['available']=self.available(m); visible.append(o)
        return {'schema_version':VERSION,'synthetic':True,'day':self.day,'ask_budget_remaining':12-self.ask_spent,
                'members':visible,'introductions':copy.deepcopy(self.introductions),
                'feedback':self.receive_feedback(),'ask_log':copy.deepcopy(self.ask_log)}
    def receive_feedback(self):
        return copy.deepcopy([e for e in self.events if e['observed_day']<=self.day])
    def resolve_asks(self,asks):
        """Each constraints bundle costs 3; each named soft field costs 1, per day budget 12.
        Declined answers stay unavailable and never get silently imputed."""
        cost=0;seen=set()
        for a in asks:
            ident=a['member_id'];field=a['field']
            if ident not in self.members or not self.available(self.members[ident]): raise ValueError('ask member')
            if field not in ['constraints']+SOFT: raise ValueError('unknown ask field')
            if (ident,field) in seen: raise ValueError('duplicate ask')
            seen.add((ident,field));cost+=3 if field=='constraints' else 1
        if cost+self.ask_spent>12: raise ValueError('ask budget exceeded')
        self.ask_spent+=cost;self.asks_total+=cost;out=[]
        for a in asks:
            m=self.members[a['member_id']]; keys=HARD if a['field']=='constraints' else [a['field']]
            for k in keys:
                if m['field_status'][k]!='declined':
                    m['fields'][k]=copy.deepcopy(m['truth'][k]);m['field_status'][k]='observed';m['field_observed_day'][k]=self.day
            out.append({'member_id':a['member_id'],'field':a['field'],'observed_day':self.day,
                        'values':{k:copy.deepcopy(m['fields'][k]) for k in keys},
                        'statuses':{k:m['field_status'][k] for k in keys}})
        self.ask_log.extend(copy.deepcopy(out));return out
    def _prob(self,a,b,shared):
        f,g=a['truth'],b['truth'];variant=self.world['variant']
        weights={'relationship_goal':.7,'relationship_pace':.4,'lifestyle':.25,'conversations':.2}
        if variant=='shift': weights={'relationship_goal':.25,'relationship_pace':.8,'lifestyle':-.25,'conversations':.5}
        fit=sum(w*(1 if f[k]==g[k] else -.5) for k,w in weights.items())
        drift=-.5 if variant=='drift' and self.day>=35 else 0
        z=-.25+a['bias']+fit+shared+drift
        return 1/(1+math.exp(-z))
    def advance(self,pairs):
        used=set()
        for pair in pairs:
            if len(pair)!=2: raise ValueError('pair length')
            a,b=pair
            if a not in self.members or b not in self.members: raise ValueError('unknown member')
            if a in used or b in used: raise ValueError('overlapping allocation')
            used.update(pair)
            if not all(self.available(self.members[i]) for i in pair): raise ValueError('unavailable member')
            if tuple(sorted(pair)) in self.past: raise ValueError('repeat pair')
            if eligibility(self.members[a],self.members[b])['status']!='feasible': raise ValueError('hard constraint violation or missing constraint')
        for a,b in sorted([sorted(p) for p in pairs]):
            self.past.add((a,b));ma,mb=self.members[a],self.members[b]
            r=rand_for(self.world['seed'],a,b,self.day)
            mid='intro_'+hashlib.sha256(f'{a}|{b}|{self.day}'.encode()).hexdigest()[:24]
            self.introductions.append({'introduction_id':mid,'user_a':a,'user_b':b,'assigned_day':self.day,'response_deadline_day':self.day+7,'logging_policy':'baseline_or_submitted_policy','propensity':None})
            shared=r.gauss(0,.45);responded=[];accept=[];response_days=[]
            for actor,other in ((ma,mb),(mb,ma)):
                responded.append(r.random()<actor['response_rate'])
                accept.append(r.random()<self._prob(actor,other,shared))
                delay=r.randint(1,7);response_days.append(self.day+delay)
                self.events.append({'introduction_id':mid,'member_id':actor['member_id'],'event':'introduction_response',
                    'value':('yes' if accept[-1] else 'no') if responded[-1] else None,
                    'missing_reason':None if responded[-1] else 'no_response',
                    'occurred_day':self.day+delay if responded[-1] else None,
                    'observed_day':self.day+delay if responded[-1] else self.day+7})
            for i in (a,b): self.busy[i]=self.day+8
            if all(responded) and all(accept):
                date_day=max(response_days)+r.randint(1,14)
                if self.world['variant']=='delayed': date_day+=r.randint(5,12)
                happened=r.random()<.78
                self.events.append({'introduction_id':mid,'member_id':None,'event':'date_happened',
                    'value':happened,'occurred_day':date_day,'observed_day':date_day})
                seconds=[];on_time=[]
                for i in (a,b): self.busy[i]=date_day+6
                if happened:
                    for actor,other in ((ma,mb),(mb,ma)):
                        answer=r.random()<actor['response_rate'];delay=r.randint(1,5)
                        p=1/(1+math.exp(-(.15+actor['second_bias']+shared+.4*(actor['truth']['relationship_goal']==other['truth']['relationship_goal']))))
                        yes=r.random()<p;seconds.append(answer and yes);on_time.append(delay<=3)
                        self.events.append({'introduction_id':mid,'member_id':actor['member_id'],'event':'second_meeting_intention',
                            'value':('yes' if yes else 'no') if answer else None,
                            'missing_reason':None if answer else 'no_response',
                            'occurred_day':date_day+delay if answer else None,'observed_day':date_day+delay if answer else date_day+5})
                    if all(seconds):
                        # No future information in availability: retirement only takes effect after both answers.
                        retire_day=max(e['observed_day'] for e in self.events if e['introduction_id']==mid)
                        self.events.append({'introduction_id':mid,'member_id':None,'event':'pause_after_mutual_interest','value':True,
                            'occurred_day':retire_day,'observed_day':retire_day})
        self.day+=1;self.ask_spent=0
    def metrics(self):
        events=self.receive_feedback();mutual=dates=msmi=0;missing=0
        for intro in self.introductions:
            es=[e for e in events if e['introduction_id']==intro['introduction_id']]
            rs=[e for e in es if e['event']=='introduction_response'];mutual+=len(rs)==2 and all(e['value']=='yes' for e in rs)
            ds=[e for e in es if e['event']=='date_happened' and e['value'] is True];dates+=bool(ds)
            ss=[e for e in es if e['event']=='second_meeting_intention']
            msmi+=bool(ds) and ds[0]['occurred_day']-intro['assigned_day']<=30 and len(ss)==2 and all(e['value']=='yes' and e['occurred_day']-ds[0]['occurred_day']<=3 for e in ss)
            missing+=sum(e.get('missing_reason')=='no_response' for e in es)
        denominator=sum(m['arrived_day']<=self.day for m in self.members.values())
        return {'assignments':len(self.introductions),'mutual_acceptances':mutual,'dates':dates,
            'mutual_second_meeting_intention':msmi,'missing_feedback':missing,'ask_cost':self.asks_total,
            'arrived_members':denominator,'msmi_per_100_arrived_members':100*msmi/max(1,denominator),
            'status':'simulated_not_observed_product_performance'}



def baseline_asks(state):
    eligible=[m for m in state['members'] if m['available'] and any(m['fields'][k] is None for k in HARD)
        and not any(m['field_status'][k]=='declined' for k in HARD)]
    return [{'member_id':m['member_id'],'field':'constraints'} for m in eligible[:state['ask_budget_remaining']//3]]




def baseline_match(state):
    members=[m for m in state['members'] if m['available']]
    past={tuple(sorted((i['user_a'],i['user_b']))) for i in state['introductions']}
    edges=[]
    for a,b in itertools.combinations(members,2):
        key=tuple(sorted((a['member_id'],b['member_id'])))
        if key in past or eligibility(a,b)['status']!='feasible':continue
        score=sum(a['fields'].get(k) is not None and a['fields'].get(k)==b['fields'].get(k) for k in SOFT)
        edges.append((-score,key))
    used=set();out=[]
    for _,pair in sorted(edges):
        if not used.intersection(pair):out.append(list(pair));used.update(pair)
    return out



def rollout(world,days=60):
    sim=Simulator(world)
    for _ in range(days):
        sim.resolve_asks(baseline_asks(sim.observe()))
        sim.advance(baseline_match(sim.observe()))
    return sim



if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=20260926);p.add_argument('-n',type=int,default=200)
    p.add_argument('--days',type=int,default=60);a=p.parse_args()
    s=rollout(generate(a.seed,a.n),a.days)
    for _ in range(40):s.advance([])
    print(json.dumps(s.metrics(),indent=2))
