from chromosome_complex import (
    load_tasks_fast, 
    create_optimized_mapping,
    ChromosomeFast,
    decode_chromosome_fast
)

# Charger les données
tasks = load_tasks_fast("tasks_large.json")
if not tasks:
    print("❌ Impossible de charger tasks_large.json")
    exit()

print(f"✓ Jobs: {len(tasks)}")

# Créer le mapping
task_by_id, job_ids, job_op_counts = create_optimized_mapping(tasks)
print(f"✓ Opérations par job chargées")

# Créer un chromosome de test
base_sequence = []
for job_id, count in job_op_counts.items():
    base_sequence.extend([job_id] * count)

# Tester la validité
print("\n🔍 Test de validation:")
chrom = ChromosomeFast(base_sequence, len(tasks))
print(f"  Chromosome initial valide: {chrom.is_valid(job_op_counts)}")

# Tester le décodage
print("\n🔍 Test de décodage:")
makespan = decode_chromosome_fast(chrom, tasks, task_by_id)
print(f"  Makespan initial: {makespan}")

# Tester avec un chromosome invalide (ajouter un gène extra)
invalid_genes = base_sequence + [job_ids[0]]  # Ajouter un job en trop
invalid_chrom = ChromosomeFast(invalid_genes, len(tasks))
print(f"\n  Chromosome avec gène extra valide: {invalid_chrom.is_valid(job_op_counts)}")

# Tester la réparation
repaired = invalid_chrom.repair(job_op_counts)
print(f"  Après réparation valide: {repaired.is_valid(job_op_counts)}")
print(f"  Longueur originale: {len(invalid_genes)}")
print(f"  Longueur réparée: {len(repaired.genes)}")

print("\n✅ Tests terminés!")