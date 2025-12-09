# main_complex.py
import time
import os
import sys

from chromosome_complex import plot_convergence_fast, plot_gantt_simplified, save_stats_fast

def main():
    print("\n" + "="*80)
    print("PROJET JSSP - ALGORITHME GÉNÉTIQUE POUR GRANDS DATASETS")
    print("="*80)
    
    # Importer la version complexe
    try:
        from chromosome_complex import (
            load_tasks_fast, 
            genetic_algorithm_advanced,
            plot_convergence_fast,
            plot_gantt_simplified,
            save_stats_fast
        )
    except ImportError:
        print("❌ Impossible d'importer chromosome_complex.py")
        print("   Assurez-vous qu'il est dans le même dossier")
        return
    
    # Vérifier data/
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"❌ Dossier '{data_dir}' manquant!")
        return
    
    # Lister fichiers
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    if not json_files:
        print("❌ Aucun fichier JSON trouvé")
        return
    
    print("\n📁 Fichiers disponibles:")
    for i, f in enumerate(json_files, 1):
        print(f"  {i}. {f}")
    
    print(f"  {len(json_files)+1}. Analyser avant exécution")
    print(f"  {len(json_files)+2}. Quitter")
    
    choice = input("\nVotre choix: ").strip()
    
    if choice == str(len(json_files)+2) or choice.lower() in ['q', 'quit']:
        return
    
    if choice == str(len(json_files)+1):
        analyze_datasets(json_files)
        return
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(json_files):
            selected_file = json_files[idx]
            run_advanced_analysis(selected_file)
        else:
            print("❌ Choix invalide")
    except ValueError:
        print("❌ Entrée invalide")

def analyze_datasets(files):
    """Analyse les datasets pour recommander des paramètres"""
    print("\n" + "="*80)
    print("ANALYSE DES DATASETS")
    print("="*80)
    
    for filename in files:
        try:
            from chromosome_complex import load_tasks_fast
            tasks = load_tasks_fast(filename)
            
            if not tasks:
                print(f"❌ {filename}: impossible de charger")
                continue
            
            total_ops = sum(len(task['operations']) for task in tasks)
            machines = set()
            for task in tasks:
                for op in task['operations']:
                    machines.add(op['machine_id'])
            
            # Recommandations
            if total_ops <= 1000:
                rec = "FAIBLE - Utiliser chromosome.py"
                pop, gens = 50, 80
            elif total_ops <= 5000:
                rec = "MOYEN - Utiliser chromosome_complex.py"
                pop, gens = 60, 100
            elif total_ops <= 20000:
                rec = "ÉLEVÉ - Utiliser chromosome_complex.py avec paramètres réduits"
                pop, gens = 40, 60
            else:
                rec = "TRÈS ÉLEVÉ - Nécessite échantillonnage ou cluster"
                pop, gens = 20, 30
            
            print(f"\n📊 {filename}:")
            print(f"   Jobs: {len(tasks)}")
            print(f"   Opérations: {total_ops}")
            print(f"   Machines: {len(machines)}")
            print(f"   Recommandation: {rec}")
            print(f"   Paramètres suggérés: pop={pop}, gens={gens}")
            
        except Exception as e:
            print(f"❌ Erreur analyse {filename}: {str(e)}")

def run_advanced_analysis(selected_file):
    """Exécute l'analyse avancée"""
    print(f"\n{'='*80}")
    print(f"ANALYSE AVANCÉE: {selected_file}")
    print(f"{'='*80}")
    
    from chromosome_complex import load_tasks_fast, genetic_algorithm_advanced
    
    # Charger
    tasks = load_tasks_fast(selected_file)
    if not tasks:
        return
    
    print(f"✓ Jobs chargés: {len(tasks)}")
    
    # Calculer stats
    total_ops = sum(len(task['operations']) for task in tasks)
    unique_machines = set()
    max_ops_per_job = 0
    min_ops_per_job = float('inf')
    
    for task in tasks:
        ops = len(task['operations'])
        max_ops_per_job = max(max_ops_per_job, ops)
        min_ops_per_job = min(min_ops_per_job, ops)
        for op in task['operations']:
            unique_machines.add(op['machine_id'])
    
    print(f"✓ Opérations totales: {total_ops}")
    print(f"✓ Machines uniques: {len(unique_machines)}")
    print(f"✓ Opérations par job: {min_ops_per_job}-{max_ops_per_job}")
    
    # Estimer complexité
    complexity_factor = total_ops * len(unique_machines) / 1000
    if complexity_factor < 10:
        complexity = "FAIBLE"
        est_time = "1-5 minutes"
    elif complexity_factor < 50:
        complexity = "MOYENNE"
        est_time = "5-15 minutes"
    elif complexity_factor < 200:
        complexity = "ÉLEVÉE"
        est_time = "15-30 minutes"
    else:
        complexity = "TRÈS ÉLEVÉE"
        est_time = "30+ minutes"
    
    print(f"✓ Complexité estimée: {complexity}")
    print(f"✓ Temps estimé: {est_time}")
    
    # Demander paramètres
    print(f"\n⚙️  Paramètres d'exécution:")
    
    # Paramètres adaptatifs
    if total_ops <= 1000:
        default_pop, default_gens = 60, 100
    elif total_ops <= 5000:
        default_pop, default_gens = 50, 80
    elif total_ops <= 20000:
        default_pop, default_gens = 40, 60
    else:
        default_pop, default_gens = 30, 40
    
    print(f"   Valeurs par défaut: Population={default_pop}, Générations={default_gens}")
    
    use_default = input(f"   Utiliser les valeurs par défaut? (o/n): ").strip().lower()
    
    if use_default in ['o', 'oui', 'y', 'yes']:
        population = default_pop
        generations = default_gens
    else:
        try:
            population = int(input(f"   Population (5-100, défaut {default_pop}): ") or default_pop)
            generations = int(input(f"   Générations (10-200, défaut {default_gens}): ") or default_gens)
            population = max(5, min(100, population))
            generations = max(10, min(200, generations))
        except:
            print("⚠️  Valeurs invalides, utilisation des valeurs par défaut")
            population = default_pop
            generations = default_gens
    
    print(f"\n✅ Configuration finale:")
    print(f"   Fichier: {selected_file}")
    print(f"   Population: {population}")
    print(f"   Générations: {generations}")
    print(f"   Taux croisement: 0.85")
    print(f"   Taux mutation: 0.15")
    print(f"   Élitisme: 5")
    
    # Confirmation pour datasets très grands
    if total_ops > 10000:
        print(f"\n⚠️  ATTENTION: Dataset très grand!")
        print(f"   Cette exécution peut prendre du temps.")
        confirm = input("   Continuer? (oui/NON): ").strip().lower()
        if confirm != 'oui':
            print("❌ Annulé")
            return
    
    # Exécuter
    print(f"\n{'='*80}")
    print("🚀 DÉMARRAGE DE L'ALGORITHME AVANCÉ")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        best_solution, best_history, avg_history, job_ids = genetic_algorithm_advanced(
            tasks=tasks,
            population_size=population,
            num_generations=generations,
            crossover_rate=0.85,
            mutation_rate=0.15,
            elitism_count=5,
            adaptive_params=True
        )
        
        total_time = time.time() - start_time
        mins = int(total_time // 60)
        secs = total_time % 60
        
        print(f"\n{'='*80}")
        print("📊 RÉSULTATS AVANCÉS")
        print(f"{'='*80}")
        print(f"⏱️  Temps total: {mins} min {secs:.1f} sec")
        print(f"🎯 Makespan final: {best_solution.makespan:.1f}")
        print(f"📈 Fitness final: {best_solution.fitness:.6f}")
        
        # Générer nom de base
        base_name = os.path.splitext(selected_file)[0]
        
        # Sauvegarder
        print(f"\n💾 Sauvegarde des résultats...")
        save_stats_fast(best_solution, tasks, f'{base_name}_advanced_stats.txt')
        
        # Visualisations adaptatives
        if total_ops <= 5000:
            print(f"\n🎨 Génération des visualisations...")
            plot_convergence_fast(best_history, avg_history, f'{base_name}_advanced_convergence.png')
            
            if total_ops <= 2000:
                plot_gantt_simplified(best_solution, tasks, job_ids, f'{base_name}_advanced_gantt.png', max_jobs=50)
            else:
                plot_gantt_simplified(best_solution, tasks, job_ids, f'{base_name}_advanced_gantt.png', max_jobs=30)
        else:
            print(f"\n🎨 Génération du graphique de convergence...")
            plot_convergence_fast(best_history, avg_history, f'{base_name}_advanced_convergence.png')
            print("⚠️  Gantt simplifié ignoré (dataset trop grand)")
        
        # Résumé
        print(f"\n{'='*80}")
        print("📁 FICHIERS GÉNÉRÉS")
        print(f"{'='*80}")
        print(f"Dossier: 'Results/'")
        print(f"• {base_name}_advanced_stats.txt - Statistiques")
        print(f"• {base_name}_advanced_convergence.png - Convergence")
        if total_ops <= 5000:
            print(f"• {base_name}_advanced_gantt.png - Gantt simplifié")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Interrompu par l'utilisateur")
        exec_time = time.time() - start_time
        print(f"⏱️  Temps écoulé: {exec_time:.1f} sec")
    except Exception as e:
        print(f"\n\n❌ ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()