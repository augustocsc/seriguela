import json
import glob

def main():
    print(f"{'Algorithm'.ljust(15)} | {'Problem'.ljust(10)} | {'Seed'.ljust(4)} | {'Best R2'.ljust(10)} | {'Top Expr'.ljust(50)}")
    print("-" * 100)
    
    files = glob.glob('results/pre_phase__t6_20260310_023310-teste/*.json')
    files.sort()
    
    for f in files:
        with open(f, 'r') as file:
            data = json.load(file)
            algo = data.get('algorithm', '')
            prob = data.get('problem', '')
            seed = str(data.get('seeds', [''])[0])
            best_r2_val = round(data.get('max_train_r2', 0), 4)
            best_r2 = str(best_r2_val)
            
            # Format negative R2 to easily spot collapse vs poor learning
            if best_r2_val < -100:
                best_r2 = "Collapse"
            elif best_r2_val < 0:
                best_r2 = f"{best_r2_val:.2f}"
            
            best_expr = ""
            if 'individual_results' in data and len(data['individual_results']) > 0:
                best_expr = data['individual_results'][0].get('best_expression', '')[:50]
                
            print(f"{algo.ljust(15)} | {prob.ljust(10)} | {seed.ljust(4)} | {best_r2.ljust(10)} | {best_expr.ljust(50)}")

if __name__ == '__main__':
    main()
