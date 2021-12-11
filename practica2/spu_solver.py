import sys
import msat_runner
import wcnf
        

        

def read(path):

    formula = wcnf.WCNFFormula()

    list_literal = {}
    with open(path, "r") as _file:
        optim_result = _file.readline().strip("\n").split(" ")[2]
        formula.header = _file.readline().strip("\n").split(" ")
        for line in _file:
            row = line.strip("\n").split(" ")
            if row[0] == "n":
                n = formula.new_var()
                formula.add_clause([n],weight=1)
                list_literal[row[1]] = n
            else:
                new_literal = []
                for literal in row[1:]:
                    new_literal.append(list_literal[literal])
                formula.add_clause(new_literal,wcnf.TOP_WEIGHT)
    return formula
    

if __name__ == "__main__":
    if len(sys.argv) == 3:
        sat_solver = msat_runner.MaxSATRunner(sys.argv[1])
        formula = read(sys.argv[2])
        print("Hard:\n",formula.hard)
        print("Soft:\n",formula.soft)
        
    else:
        print("Usage: {} <MaxSat> <problem to solve>".format(sys.argv[0]))
