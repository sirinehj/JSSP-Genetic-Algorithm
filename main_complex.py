import time
import os

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
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("   Assurez-vous que chromosome_complex.py est dans le même dossier")
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
    
    try:
        choice = input("\nVotre choix (numéro): ").strip()
        idx = int(choice) - 1
        
        if 0 <= idx < len(json_files):
            selected_file = json_files[idx]
            print(f"\n✅ Fichier sélectionné: {selected_file}")
            run_advanced_analysis(selected_file, 
                                  load_tasks_fast,
                                  genetic_algorithm_advanced,
                                  plot_convergence_fast,
                                  plot_gantt_simplified,
                                  save_stats_fast)
        else:
            print("❌ Choix invalide")
    except ValueError:
        print("❌ Entrée invalide. Veuillez entrer un numéro.")
    except KeyboardInterrupt:
        print("\n⏹️  Interrompu par l'utilisateur")

def run_advanced_analysis(selected_file, 
                         load_tasks_fast,
                         genetic_algorithm_advanced,
                         plot_convergence_fast,
                         plot_gantt_simplified,
                         save_stats_fast):
    """Exécute l'analyse avancée"""
    print(f"\n{'='*80}")
    print(f"ANALYSE AVANCÉE: {selected_file}")
    print(f"{'='*80}")
    
    # Charger
    tasks = load_tasks_fast(selected_file)
    if not tasks:
        print("❌ Impossible de charger les tâches")
        return
    
    print(f"✓ Jobs chargés: {len(tasks)}")
    
    # Calculer stats
    total_ops = sum(len(task['operations']) for task in tasks)
    print(f"✓ Opérations totales: {total_ops}")
    
    # Paramètres adaptatifs
    if total_ops <= 1000:
        default_pop, default_gens = 60, 100
        mode = "FAIBLE"
    elif total_ops <= 5000:
        default_pop, default_gens = 50, 80
        mode = "MOYEN"
    elif total_ops <= 20000:
        default_pop, default_gens = 40, 60
        mode = "ÉLEVÉ"
    else:
        default_pop, default_gens = 25, 30
        mode = "TRÈS ÉLEVÉ"
    
    print(f"\n⚙️  Mode: {mode}")
    print(f"   Population par défaut: {default_pop}")
    print(f"   Générations par défaut: {default_gens}")
    
    # Utiliser les valeurs par défaut pour commencer
    population = default_pop
    generations = default_gens
    
    print(f"\n✅ Configuration:")
    print(f"   Fichier: {selected_file}")
    print(f"   Population: {population}")
    print(f"   Générations: {generations}")
    print(f"   Taux croisement: 0.85")
    print(f"   Taux mutation: 0.15")
    print(f"   Élitisme: 5")
    
    # Confirmation pour datasets très grands
    if total_ops > 10000:
        print(f"\n⚠️  ATTENTION: Dataset très grand ({total_ops} opérations)!")
        print(f"   Cette exécution peut prendre du temps.")
        confirm = input("   Continuer? (o/n): ").strip().lower()
        if confirm not in ['o', 'oui', 'y', 'yes']:
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
        print(f"\n🎨 Génération des visualisations...")
        plot_convergence_fast(best_history, avg_history, f'{base_name}_advanced_convergence.png')
        
        if total_ops <= 5000:
            plot_gantt_simplified(best_solution, tasks, job_ids, 
                                 f'{base_name}_advanced_gantt.png', 
                                 max_jobs=30 if total_ops > 2000 else 50)
        else:
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