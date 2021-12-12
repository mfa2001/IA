import sys
import msat_runner
import wcnf
        
class SPUFormula():
    def __init__(self):
        self.soft = []
        self.hard_dep = []
        self.hard_conf = []
        self.header = []
        self.optim = 0
        
def read(path):

    spuForm = SPUFormula()

    list_literal = {}
    i = 1
    with open(path, "r") as _file:
        spuForm.optim = _file.readline().strip("\n").split(" ")[2]
        spuForm.header = _file.readline().strip("\n").split(" ")
        for line in _file:
            row = line.strip("\n").split(" ")
            literal = []
            if row[0] == "n":
                list_literal[row[1]] = i
                i+=1
                spuForm.soft.append(list_literal[row[1]])
            elif row[0] == "d":
                spuForm.hard_dep.append(create_clause(row[1:],list_literal))
            else:
                spuForm.hard_conf.append(create_clause(row[1:],list_literal))

    dictionary = {}
    for pkg in list_literal:
        dictionary[list_literal[pkg]] = pkg
    return spuForm,dictionary

def transform_formula(problem: SPUFormula):

    form = wcnf.WCNFFormula()
    highest_var = max(abs(l) for l in problem.soft)
    while form.num_vars < highest_var:
             form.new_var()
    for c in problem.soft:
        form.add_clause([c],weight=1)
    
    for c in problem.hard_dep:
        new_literal = []
        new_literal.append(-c[0])
        for clausul in c[1:]:
            new_literal.append(clausul)
        form.add_clause(new_literal,wcnf.TOP_WEIGHT)


    for c in problem.hard_conf:
        new_literal = []
        for clausul in c:
            new_literal.append(-clausul)
        form.add_clause(new_literal,wcnf.TOP_WEIGHT)
    return form
        


def create_clause(clausule,list_lit):
    literal = []
    for pkg in clausule:
        literal.append(list_lit[pkg])
    return literal


if __name__ == "__main__":
    if len(sys.argv) == 3:
        sat_solver = msat_runner.MaxSATRunner(sys.argv[1])
        problem,dictionary = read(sys.argv[2])
        #print("soft:\n",problem.soft,"\nhard_dep:\n",problem.hard_dep,"\nHard_conf:\n",problem.hard_conf)
        transformed_problem = transform_formula(problem)
        #print("Soft:\n",transformed_problem.soft,"\nHard:\n",transformed_problem.hard)
        opt,model = sat_solver.solve(transformed_problem)
        #print(opt,model)
        number_not_installed = 0
        not_installed = []

        for n in model:
            if n < 0:
                not_installed.append(dictionary[abs(n)])
                number_not_installed+=1
        
        print("o ",number_not_installed)
        print("v ", '--'.join([v for v in not_installed]))


    else:
        print("Usage: {} <MaxSat> <problem to solve>".format(sys.argv[0]))
