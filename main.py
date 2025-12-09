# main.py
import time
from Chromosome import genetic_algorithm, load_tasks, plot_convergence, plot_gantt_chart, save_solution_stats

def main():
    # Enregistrer le temps de début
    start_time = time.time()
    
    # Charger les données
    print("Chargement des tâches...")
    
    # D'abord essayer avec le fichier de test
    try:
        tasks = load_tasks('tasks_small.json')
        print(f"✓ Fichier de test chargé: {len(tasks)} tâches")
    except:
        tasks = load_tasks('tasks.json')
        print(f"✓ Fichier principal chargé: {len(tasks)} tâches")
    
    # Paramètres de l'algorithme génétique
    POPULATION_SIZE = 30      # Réduit pour tester
    CROSSOVER_RATE = 0.8      # Taux de croisement
    MUTATION_RATE = 0.1       # Taux de mutation
    NUM_GENERATIONS = 50      # Nombre réduit pour tester
    ELITISM_COUNT = 2         # Conserver les 2 meilleurs
    
    print(f"\nConfiguration de l'algorithme génétique:")
    print(f"  Population: {POPULATION_SIZE}")
    print(f"  Générations: {NUM_GENERATIONS}")
    print(f"  Taux croisement: {CROSSOVER_RATE}")
    print(f"  Taux mutation: {MUTATION_RATE}")
    print(f"  Élitisme: {ELITISM_COUNT}")
    
    # Exécuter l'algorithme génétique
    print("\n" + "="*60)
    print("EXÉCUTION DE L'ALGORITHME GÉNÉTIQUE")
    print("="*60)
    
    try:
        best_solution, best_fitness_history, avg_fitness_history, job_ids = genetic_algorithm(
            tasks=tasks,
            population_size=POPULATION_SIZE,
            crossover_rate=CROSSOVER_RATE,
            mutation_rate=MUTATION_RATE,
            num_generations=NUM_GENERATIONS,
            elitism_count=ELITISM_COUNT
        )
        
        # Calculer le temps d'exécution
        execution_time = time.time() - start_time
        minutes = int(execution_time // 60)
        seconds = execution_time % 60
        
        print(f"\n{'='*60}")
        print("VISUALISATION DES RÉSULTATS")
        print(f"{'='*60}")
        print(f"Temps d'exécution: {minutes} min {seconds:.1f} sec")
        
        # Sauvegarder les statistiques
        save_solution_stats(best_solution, tasks, 'solution_stats.txt')
        
        # Visualiser la solution (Gantt)
        print("\nGénération du diagramme de Gantt...")
        plot_gantt_chart(best_solution, tasks, job_ids, 'gantt_chart_jssp.png')
        
        # Visualiser la convergence
        print("\nGénération du graphique de convergence...")
        plot_convergence(best_fitness_history, avg_fitness_history, 
                        'convergence_jssp.png')
        
        # Résumé final
        print(f"\n{'='*60}")
        print("RÉSUMÉ FINAL")
        print(f"{'='*60}")
        print(f"✓ Meilleur makespan: {best_solution.makespan:.2f}")
        print(f"✓ Fitness: {best_solution.fitness:.6f}")
        print(f"✓ Fichiers générés:")
        print(f"  - gantt_chart_jssp.png (Diagramme de Gantt)")
        print(f"  - convergence_jssp.png (Convergence)")
        print(f"  - solution_stats.txt (Statistiques)")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\n\n❌ Exécution interrompue par l'utilisateur")
        execution_time = time.time() - start_time
        print(f"Temps écoulé: {execution_time:.1f} secondes")
    except Exception as e:
        print(f"\n\n❌ ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
        execution_time = time.time() - start_time
        print(f"Temps écoulé avant l'erreur: {execution_time:.1f} secondes")

if __name__ == "__main__":
    main()