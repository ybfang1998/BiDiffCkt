import pandas as pd
SPEC_UNNORMALIZE_FACTORS = {
    "gain": 100,
    "GBW": 10e6,
    "PM": 180,
    "CMRR": 100,
    "PSRR": 100,
    "PWR": 1e-3
}

normalised_targets_external = {
    "gain": 0.8,
    "GBW": 1.0,
    "PM": 0.31,
    "CMRR": 0.6,
    "PSRR": 0.7,
    "PWR": 0.2
}

normalised_targets_high = {
    "gain": 0.6,
    "GBW": 0.7,
    "PM": 0.31,
    "CMRR": 0.53,
    "PSRR": 0.53,
    "PWR": 0.35
}

normalised_targets_med = {
    "gain": 0.53,
    "GBW": 0.4,
    "PM": 0.28,
    "CMRR": 0.37,
    "PSRR": 0.37,
    "PWR": 0.65
}

normalised_targets_low = {
    "gain": 0.4,
    "GBW": 0.1,
    "PM": 0.25,
    "CMRR": 0.2,
    "PSRR": 0.2,
    "PWR": 1.0
}

def Diffckt_circuit_eval(results_csv_path, spec_level="external"):
    df = pd.read_csv(results_csv_path)
    if spec_level == "external":
        targets = normalised_targets_external
    elif spec_level == "high":
        targets = normalised_targets_high
    elif spec_level == "medium":
        targets = normalised_targets_med
    elif spec_level == "low":
        targets = normalised_targets_low
    else:
        raise ValueError("Invalid spec_level. Must be 'external', 'high', 'medium', or 'low'.")

    n_sat = 0
    max_fom = 0
    avg_fom = 0
    for index, row in df.iterrows():
        fom = row['fom']
        if fom > 0:
            n_sat+=1
            avg_fom += fom
        if fom > max_fom:
            max_fom = fom
    # print("N of circuit satisfied:", n_sat)
    # print("max fom:", max_fom)
    # print("avg_fom2:", avg_fom/100)
    return n_sat, round(max_fom, 2), round(avg_fom/100, 2)



def ocb_circuit_eval(results_csv_path):
    fom_results = pd.read_csv(results_csv_path)
    # Rate of generated circuit that could achieve the specification
    achievement_count = len(fom_results[fom_results['fom'] > 0])
    # print(achievement_count)

    # Best/average FoM Results:
    valid_fom_results = fom_results[fom_results['fom'] > 0]["fom"]
    # print(f"Best FOM: {round(valid_fom_results.max(), 2)}")
    # print(f"Average FOM among success results: {round(valid_fom_results.sum()/50, 2)}")

    return achievement_count, round(valid_fom_results.max(), 2), round(valid_fom_results.sum()/50, 2)