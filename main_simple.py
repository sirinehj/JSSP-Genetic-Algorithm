# main_simple.py (version avec Gantt amélioré)
import time
from Chromosome import load_tasks, genetic_algorithm, plot_gantt_chart, plot_convergence, save_solution_stats
import os

def main():
    print("\n" + "="*70)
    print("PROJET JSSP - ALGORITHME GÉNÉTIQUE")
    print("="*70)
    
    # Dossier de données
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"❌ Dossier '{data_dir}' non trouvé!")
        print(f"   Création du dossier...")
        os.makedirs(data_dir)
        print(f"✓ Dossier '{data_dir}' créé")
        print(f"   Veuillez placer vos fichiers JSON dans ce dossier")
        return
    
    # Chercher un fichier spécifique
    target_file = "tasks_small.json"
    
    print(f"\n🔍 Recherche du fichier: {data_dir}/{target_file}")
    
    if not os.path.exists(os.path.join(data_dir, target_file)):
        print(f"❌ Fichier '{target_file}' non trouvé!")
        print(f"\n📁 Fichiers disponibles dans '{data_dir}/':")
        
        json_files = []
        for file in os.listdir(data_dir):
            if file.endswith('.json'):
                json_files.append(file)
                print(f"   • {file}")
        
        if json_files:
            print(f"\n⚠️  Utilisation du premier fichier trouvé: {json_files[0]}")
            target_file = json_files[0]
        else:
            print("❌ Aucun fichier JSON trouvé!")
            return
    
    # Charger les données
    print(f"\n📁 Chargement: {data_dir}/{target_file}")
    tasks = load_tasks(target_file)
    
    if not tasks:
        print(f"❌ Impossible de charger les données")
        return
    
    print(f"✓ Jobs chargés: {len(tasks)}")
    
    # Analyser
    total_ops = sum(len(task['operations']) for task in tasks)
    print(f"✓ Opérations totales: {total_ops}")
    
    # Paramètres selon la taille
    if total_ops <= 50:
        population = 20
        generations = 30
        mode = "TEST RAPIDE"
    elif total_ops <= 200:
        population = 30
        generations = 50
        mode = "PETIT DATASET"
    elif total_ops <= 1000:
        population = 50
        generations = 80
        mode = "DATASET MOYEN"
    elif total_ops <= 5000:
        population = 70
        generations = 100
        mode = "GRAND DATASET"
    else:
        population = 80
        generations = 120
        mode = "TRÈS GRAND DATASET"
    
    print(f"\n⚙️  Mode: {mode}")
    print(f"   Population: {population}")
    print(f"   Générations: {generations}")
    
    # Exécuter
    print(f"\n{'='*70}")
    print("🚀 DÉMARRAGE DE L'OPTIMISATION")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        best_solution, best_history, avg_history, job_ids = genetic_algorithm(
            tasks=tasks,
            population_size=population,
            crossover_rate=0.85,
            mutation_rate=0.15,
            num_generations=generations,
            elitism_count=5
        )
        
        # Temps d'exécution
        exec_time = time.time() - start_time
        mins = int(exec_time // 60)
        secs = exec_time % 60
        
        print(f"\n{'='*70}")
        print("📊 RÉSULTATS")
        print(f"{'='*70}")
        print(f"⏱️  Temps: {mins} min {secs:.1f} sec")
        print(f"🎯 Makespan: {best_solution.makespan:.2f}")
        print(f"📈 Fitness: {best_solution.fitness:.6f}")
        
        # Générer un nom de base
        base_name = os.path.splitext(target_file)[0]
        
        # Sauvegarder
        print(f"\n💾 Sauvegarde des résultats...")
        save_solution_stats(best_solution, tasks, f'{base_name}_stats.txt')
        
        # Visualisations adaptatives
        if total_ops <= 1000:  # Augmenté à 1000
            print(f"\n🎨 Génération des graphiques...")
            plot_gantt_chart(best_solution, tasks, job_ids, f'{base_name}_gantt.png')
            plot_convergence(best_history, avg_history, f'{base_name}_convergence.png')
        elif total_ops <= 5000:
            print(f"\n🎨 Génération du graphique de convergence...")
            plot_convergence(best_history, avg_history, f'{base_name}_convergence.png')
            print("⚠️  Diagramme de Gantt ignoré (trop de données pour une visualisation claire)")
        else:
            print(f"\n⚠️  Visualisations ignorées (dataset trop grand)")
        
        # Résumé
        print(f"\n{'='*70}")
        print("📁 FICHIERS GÉNÉRÉS DANS 'Results/'")
        print(f"{'='*70}")
        print(f"• {base_name}_stats.txt - Statistiques détaillées")
        if total_ops <= 1000:
            print(f"• {base_name}_gantt.png - Diagramme de Gantt")
            print(f"• {base_name}_convergence.png - Graphique de convergence")
        elif total_ops <= 5000:
            print(f"• {base_name}_convergence.png - Graphique de convergence")
        print(f"{'='*70}")
        
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