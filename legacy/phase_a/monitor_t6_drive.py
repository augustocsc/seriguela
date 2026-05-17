import os
import glob
import json
import time

def parse_aggregate_json(filepath):
    """Lê o JSON final e extrai métricas importantes."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        algo = data.get('algorithm', 'Unknown')
        prob = data.get('problem', 'Unknown')
        seed = str(data.get('seeds', [''])[0])
        max_r2 = data.get('max_train_r2', 0)
        
        # Pega a melhor expressão
        best_expr = ""
        if 'individual_results' in data and len(data['individual_results']) > 0:
            best_expr = data['individual_results'][0].get('best_expression', '')[:50]
            
        return algo, prob, seed, max_r2, best_expr
    except Exception as e:
        return None

def monitor_t6():
    print(" Buscando resultados salvos no Google Drive (Test 6)... ".center(80, "="))
    
    # Procura a pasta do Google Drive ou localmente
    search_dirs = [
        "/content/drive/MyDrive/seriguela_results",
        "../../results",
        "results"
    ]
    
    base_dir = None
    for d in search_dirs:
        if os.path.exists(d):
            base_dir = d
            break
            
    if not base_dir:
        print("❌ Não foi possível encontrar a pasta de resultados nas rotas padrão.")
        return
        
    print(f"📁 Pasta encontrada: {base_dir}\n")
    
    # Encontra as subpastas pre_phase__t6*
    t6_folders = glob.glob(os.path.join(base_dir, "pre_phase__t6_*"))
    if not t6_folders:
        print("Nenhuma pasta 'pre_phase__t6_*' encontrada ainda. O script provavelmente ainda não salvou a primeira etapa.")
        return
        
    all_results = []
    
    for folder in t6_folders:
        json_files = glob.glob(os.path.join(folder, "aggregate_*.json"))
        for jf in json_files:
            parsed = parse_aggregate_json(jf)
            if parsed:
                all_results.append(parsed)
                
    if not all_results:
        print("Nenhum arquivo JSON finalizado encontrado nas pastas. O primeiro experimento da fila ainda deve estar rodando.")
        return
        
    # Ordenar por Problema -> Algoritmo -> Seed
    all_results.sort(key=lambda x: (x[1], x[0], x[2]))
    
    print(f"{'Problema'.ljust(10)} | {'Algoritmo'.ljust(15)} | {'Seed'.ljust(4)} | {'Melhor R2'.ljust(10)} | {'Top Expressão'.ljust(40)}")
    print("-" * 85)
    
    for algo, prob, seed, r2, expr in all_results:
        # Formata o R2 para melhorar leitura de collapse vs sucesso
        if r2 < -100:
            r2_str = "Colapso"
        elif r2 < 0:
            r2_str = f"{r2:.2f}"
        else:
            r2_str = f"{r2:.4f}"
            
        print(f"{prob.ljust(10)} | {algo.ljust(15)} | {seed.ljust(4)} | {r2_str.ljust(10)} | {expr.ljust(40)}")
        
    print("-" * 85)
    print(f"\nTotal concluído: {len(all_results)} de 18 treinamentos.")

if __name__ == "__main__":
    monitor_t6()
