import os
from mai_dx import MaiDxOrchestrator
from loguru import logger

if __name__ == "__main__":
    # Ensure the debug logger writes to the correct directory if enabled
    os.environ["MAIDX_DEBUG"] = "1"

    # Example case inspired by the paper's Figure 1
    initial_info = (
        "A 29-year-old woman was admitted to the hospital because of sore throat and peritonsillar swelling "
        "and bleeding. Symptoms did not abate with antimicrobial therapy."
    )

    full_case = """
    Patient: 29-year-old female.
    History: Onset of sore throat 7 weeks prior to admission. Worsening right-sided pain and swelling.
    No fevers, headaches, or gastrointestinal symptoms. Past medical history is unremarkable.
    Physical Exam: Right peritonsillar mass, displacing the uvula. No other significant findings.
    Initial Labs: FBC, clotting studies normal.
    MRI Neck: Showed a large, enhancing mass in the right peritonsillar space.
    Biopsy (H&E): Infiltrative round-cell neoplasm with high nuclear-to-cytoplasmic ratio and frequent mitotic figures.
    Biopsy (Immunohistochemistry): Desmin and MyoD1 diffusely positive. Myogenin multifocally positive.
    Final Diagnosis from Pathology: Embryonal rhabdomyosarcoma of the pharynx.
    """

    ground_truth = "Embryonal rhabdomyosarcoma of the pharynx"

    try:
        print("\n" + "=" * 80)
        print("    MAI-DxO - AUTONOMOUS DIAGNOSIS BENCHMARK")
        print("=" * 80)

        orchestrator = MaiDxOrchestrator.create_variant(
            "no_budget",
            model_name="gpt-4o-mini",
            max_iterations=8,
            request_delay=0.5,  # Faster for demo
        )

        result = orchestrator.run(
            initial_case_info=initial_info,
            full_case_details=full_case,
            ground_truth_diagnosis=ground_truth,
        )

        print(f"\n🚀 Final Diagnosis: {result.final_diagnosis}")
        print(f"🎯 Ground Truth: {result.ground_truth}")
        print(f"⭐ Accuracy Score: {result.accuracy_score}/5.0")
        print(f"   Reasoning: {result.accuracy_reasoning}")
        print(f"💰 Total Cost: ${result.total_cost:,}")
        print(f"🔄 Iterations: {result.iterations}")
        print(f"⏱️  Mode: {orchestrator.mode}")

    except Exception as e:
        logger.exception(
            f"An error occurred during the autonomous session: {e}"
        )
        print(f"\n❌ Error occurred: {e}. Check API keys in your .env file.")
