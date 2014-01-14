import scipy.special as sp
import numpy as np
import numpy.random as rm
import matplotlib.pyplot as pl

POPULATION_SIZE = 100
SD = 200*np.sqrt(2)
K = 16
THRESHOLD = 20
SAMPLES = 1000

Q = np.log(10)/400

def gee(rd):
    return 1/(np.sqrt(1+3*Q**2*rd**2/np.pi**2))

def one_over_dee_squared(diff, rd):
    return Q**2*gee(rd)**2*glicko_winprob(diff, rd)*\
                               (1-glicko_winprob(diff, rd))

def normal_winprob(diff):
    return .5 + .5*sp.erf(diff/400)

def log_winprob(diff):
    return 1/(1+10**(-diff/400))

def glicko_winprob(diff, rd):
    return 1/(1+10**(-gee(rd)*diff/400))

winprob = log_winprob

def game_sim(diff):
    if rm.normal(diff, SD)>0:
        return 1
    else: return 0

def update_ratings(merson1, merson2, result):
    rat1 = merson1.rating
    rat2 = merson2.rating
    merson1.rating = rat1 +\
                     K*(result - winprob(rat1 - rat2))
    merson1.rating_history.append(merson1.rating)
    merson2.rating = rat2 + K*(winprob(rat1 - rat2) - result)
    merson2.rating_history.append(merson2.rating)
    return None

def update_glicko(merson1, merson2, result):
    rat1 = merson1.rating
    rat2 = merson2.rating
    rd1 = min(np.sqrt(merson1.rd**2+35), 200.0)
    rd2 = min(np.sqrt(merson2.rd**2+35), 200.0)
    diff = rat1-rat2
    merson1.rating = rat1 + Q/(1/rd1**2+one_over_dee_squared(diff, rd2))*gee(rd2)**2\
                     *(result-glicko_winprob(diff, rd2))
    merson1.rd= max(np.sqrt(1/(1/rd1**2+one_over_dee_squared(diff, rd2))), 20.0)
    merson2.rating = rat2 + Q/(1/rd2**2+one_over_dee_squared(-diff, rd1))*gee(rd1)**2\
                     *(1-result-glicko_winprob(-diff, rd1))
    merson2.rd= max(np.sqrt(1/(1/rd2**2+one_over_dee_squared(-diff, rd1))), 20.0)
    merson1.rating_history.append(merson1.rating)
    merson2.rating_history.append(merson2.rating)
    merson1.rd_history.append(merson1.rd)
    merson2.rd_history.append(merson2.rd)
    return None

update = update_glicko
    
class Merson:
    def __init__(self, rating, skill):
        self.rating = rating
        self.rating_history = [rating]
        self.skill = skill
        self.skill_history = [skill]
        self.rd = 200.0
        self.rd_history = [self.rd]

def generate_population(n):
    return [Merson(1600.0, rm.normal(1600, 100)) for i in range(n)]

def pairings(n):
    a = np.arange(n)
    rm.shuffle(a)
    return a.reshape((n/2,2))

def simulate_round(pop):
    pairs = pairings(len(pop))
    for pair in pairs:
           update(pop[pair[0]], pop[pair[1]],
                          game_sim(pop[pair[0]].skill-pop[pair[1]].skill))
##           pop[pair[0]].skill += 20*(rm.randint(2)-.5)
##           if pop[pair[0]] == meeple[0]:
##               pop[pair[0]].skill += .4
           pop[pair[0]].skill_history.append(pop[pair[0]].skill)
##           pop[pair[1]].skill += 20*(rm.randint(2)-.5)
##           if pop[pair[1]] == meeple[0]:
##               pop[pair[1]].skill += .4
           pop[pair[1]].skill_history.append(pop[pair[1]].skill)
    adjust_skills(pop)
    return None

def adjust_skills(pop):
    for p in pop:
        p.skill -= average_skill(pop)-average_rating(pop)

def average_skill(pop):
    return sum([p.skill for p in pop])/len(pop)

def average_rating(pop):
    return sum([p.rating for p in pop])/len(pop)

def regroup(e, p):
    rm.shuffle(e)
    p_new = [e.pop() for i in range(20)]
    e.sort(key=lambda x: x.rating, reverse=True)
    p.sort(key=lambda x: x.rating, reverse=True)
    for j in range(THRESHOLD):
        if len(p)>0 and p[0].rating > e[0].rating:
            p_new.append(p.pop(0))
        else:
            p_new.append(e.pop(0))
    e.extend(p)
    return e, p_new

def regroup_by_skill(e, p):
    e.sort(key=lambda x: x.skill, reverse=True)
    p.sort(key=lambda x: x.skill, reverse=True)
    e_new = []
    p_new = []
    count = 0
    for j in range(THRESHOLD):
        if len(p)>0 and p[0].skill > e[0].skill:
            p_new.append(p.pop(0))
        else:
            p_new.append(e.pop(0))
            count = count +1
    e.extend(p)
    return count

meeple = generate_population(POPULATION_SIZE)
##mexperts = []
##meeple = [Merson(1600.0, 1600.0), Merson(1600.0, 1600.0)]
adjust_skills(meeple)
for j in range(SAMPLES):
    simulate_round(meeple)
##    simulate_round(mexperts)
##    meeple, mexperts = regroup(meeple, mexperts)

#count = regroup_by_skill(meeple, mexperts)
#print count

#pl.plot(meeple[0].rating_history)
##mexperts.sort(key = lambda x: x.skill, reverse =True)
pl.plot([meeple[0].rating_history[i]-2*meeple[0].rd_history[i] for i in range(SAMPLES)])
pl.plot([meeple[0].rating_history[i]+2*meeple[0].rd_history[i] for i in range(SAMPLES)])
##pl.plot(meeple[0].rating_history)
pl.plot(meeple[0].skill_history)
#pl.plot([sum(meeple[0].rating_history[:k+1])/(k+1)\
#          for k in range(SAMPLES)])
pl.show()
