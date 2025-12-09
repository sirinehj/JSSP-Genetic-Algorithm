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
    target_file = "tasks_small.json"  # Changez ceci selon vos besoins
    
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
        print(f"\n⚙️  Mode: TEST RAPIDE")
    elif total_ops <= 200:
        population = 30
        generations = 50
        print(f"\n⚙️  Mode: PETIT DATASET")
    elif total_ops <= 1000:
        population = 50
        generations = 80
        print(f"\n⚙️  Mode: DATASET MOYEN")
    elif total_ops <= 5000:
        population = 70
        generations = 100
        print(f"\n⚙️  Mode: GRAND DATASET")
    else:
        population = 80
        generations = 120
        print(f"\n⚙️  Mode: TRÈS GRAND DATASET")
    
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
        
        # Sauvegarder
        print(f"\n💾 Sauvegarde des résultats...")
        save_solution_stats(best_solution, tasks, 'solution_stats.txt')
        
        # Visualisations (seulement pour datasets moyens/petits)
        if total_ops <= 500:
            print(f"\n🎨 Génération des graphiques...")
            plot_gantt_chart(best_solution, tasks, job_ids, 'gantt.png')
            plot_convergence(best_history, avg_history, 'convergence.png')
        elif total_ops <= 2000:
            print(f"\n🎨 Génération du graphique de convergence...")
            plot_convergence(best_history, avg_history, 'convergence.png')
            print("⚠️  Diagramme de Gantt ignoré (trop de données)")
        else:
            print(f"\n⚠️  Visualisations ignorées (dataset trop grand)")
        
        print(f"\n{'='*70}")
        print("✅ EXÉCUTION TERMINÉE")
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