"""Metropolitan distributed-AI heat-reuse screening model.

The model evaluates 120 facility-matched B300 GPU micro-data-centre sites in a
Los Angeles metropolitan module. It uses the California CZ2025 LAX EPW weather
series, explicitly tracks captured and uncaptured IT heat, process-heat reuse,
PV/T heat rejection, evaporative trim, dry rejection, auxiliary electricity,
and operational greenhouse-gas scenarios. The main routine regenerates the shipped
central, workload/sizing, laundry-heat, carbon, weather and network-site CSV tables.

The networking screen uses NVIDIA HGX B300 endpoint specifications and a
route-adjusted fibre propagation model. It does not claim application-level
training parity with a co-located cluster; commercial Spectrum-XGS capability
for geographically distributed single-job training is treated as external
technology evidence, while this code quantifies only metropolitan physical
placement and endpoint capacity.
"""
from __future__ import annotations
import argparse, io, math, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

HOURS = 8760
GPU_SYS_KW = 14.5 / 8.0
N_PER_CLASS = 20
ROUTE_FACTOR = 1.40
FIBRE_US_PER_KM = 5.0
GPU_ENDPOINT_GBPS = 800.0

EPW_URL = (
    "https://raw.githubusercontent.com/sound-data/DEER-Prototypes-EnergyPlus/"
    "7283693a59cecd08644ecc23464337033a681b90/weather/"
    "CA_CZ06_LOS-ANGELES-IAP_722950_CZ2025.epw"
)

GPU_COUNTS = {
    "Small car wash": 9,
    "Medium car wash": 15,
    "Large car wash": 36,
    "Small laundromat": 41,
    "Medium laundromat": 83,
    "Large laundromat": 124,
}
THROUGHPUT_PER_DAY = {
    "Small car wash": 75,
    "Medium car wash": 125,
    "Large car wash": 300,
    "Small laundromat": 200,
    "Medium laundromat": 400,
    "Large laundromat": 600,
}
# Scenario inputs, not measured Los Angeles averages. Sensitivities explicitly
# test the dominant laundry coefficient over 0.5x-1.25x the central value.
PROCESS_HEAT_KWH_PER_SERVICE = {
    "Small car wash": 2.88, "Medium car wash": 2.88, "Large car wash": 2.88,
    "Small laundromat": 4.96, "Medium laundromat": 4.96, "Large laundromat": 4.96,
}
CAR_SHAPE = np.array([.05,.04,.04,.04,.05,.10,.25,.55,.80,1.00,1.10,1.15,
                      1.15,1.10,1.05,1.05,1.10,1.10,.95,.65,.35,.18,.10,.07], float)
LAUNDRY_SHAPE = np.array([.12,.10,.09,.08,.10,.18,.38,.65,.85,1.00,1.05,1.08,
                          1.05,1.02,1.00,1.03,1.10,1.18,1.22,1.18,1.05,.82,.55,.28], float)
CAR_SHAPE /= CAR_SHAPE.mean(); LAUNDRY_SHAPE /= LAUNDRY_SHAPE.mean()

# Operational carbon screen.
NG_ELECTRICITY_GCO2E_PER_KWH = 486.0
PV_ELECTRICITY_GCO2E_PER_KWH = 43.0
NG_CO2_KG_PER_MMBTU_HHV = 53.06
NG_CH4_G_PER_MMBTU_HHV = 1.0
NG_N2O_G_PER_MMBTU_HHV = 0.10
GWP_CH4 = 28.0; GWP_N2O = 265.0
NG_UPSTREAM_KGCO2E_PER_MMBTU_LHV = 10.3
HHV_PER_LHV = 1.108
GAS_HEATER_EFFICIENCY = 0.85
MMBTU_PER_MWH = 3.412141633

DRY_REJECTION_THERMAL_PER_ELECTRIC = 22.2
LIQUID_LOOP_PUMP_FRAC_IT = 0.010


def read_epw(path: str | Path | None = None, url: str = EPW_URL):
    if path:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            text = response.read().decode("utf-8", errors="replace")
    rows = []
    for line in text.splitlines()[8:]:
        a = line.split(",")
        if len(a) < 24:
            continue
        try:
            rows.append((float(a[6]), float(a[8])/100.0, float(a[21]), float(a[22])/10.0, float(a[13])))
        except ValueError:
            continue
    if len(rows) != HOURS:
        raise ValueError(f"Expected 8760 EPW records; got {len(rows)}")
    arr = np.asarray(rows, float)
    return {"t_air":arr[:,0], "rh":np.clip(arr[:,1],.001,1), "wind":arr[:,2],
            "cloud":np.clip(arr[:,3],0,1), "ghi":arr[:,4], "local_hour":np.arange(HOURS)%24}


def rejection_flux(wx, ts_reject_C=45.0):
    sigma=5.670374419e-8; rho=1.18; g=9.81; stack=3.0; cd=.60; kloss=25.0
    eta=.80; cp=1005.0; open_ratio=.012; cp_shroud=.35; f_sky=.95; eps_surf=.92; pv_eff=.19
    tsK=ts_reject_C+273.15; t=wx["t_air"]; taK=t+273.15; dT=np.maximum(tsK-taK,0)
    dpb=rho*g*stack*dT/taK; ub=cd*np.sqrt(2*dpb/(rho*(1+kloss)))
    qstack=eta*rho*cp*open_ratio*ub*dT
    dpw=cp_shroud*.5*rho*np.square(wx["wind"])
    ubw=cd*np.sqrt(2*(dpb+dpw)/(rho*(1+kloss)))
    qshroud=eta*rho*cp*open_ratio*np.maximum(ubw-ub,0)*dT
    aa=17.625; bb=243.04
    gamma=np.log(wx["rh"])+aa*t/(bb+t); tdp=bb*gamma/(aa-gamma)
    eps_clear=np.clip(.711+.0056*tdp+7.3e-5*np.square(tdp)+.013,.5,1.0)
    eps_sky=eps_clear+(1-eps_clear)*wx["cloud"]; tsky=taK*np.power(eps_sky,.25)
    qrad=np.maximum(f_sky*eps_surf*sigma*(np.power(tsK,4)-np.power(tsky,4)),0)
    solar=np.maximum(wx["ghi"],0)*(.90-pv_eff)
    return np.maximum(qstack+qshroud+qrad-solar,0)


def gas_heat_lifecycle_tco2e_per_mwh_useful(efficiency=GAS_HEATER_EFFICIENCY):
    combustion=(NG_CO2_KG_PER_MMBTU_HHV+NG_CH4_G_PER_MMBTU_HHV*GWP_CH4/1000+
                NG_N2O_G_PER_MMBTU_HHV*GWP_N2O/1000)
    upstream_hhv=NG_UPSTREAM_KGCO2E_PER_MMBTU_LHV/HHV_PER_LHV
    return (combustion+upstream_hhv)*MMBTU_PER_MWH/1000/efficiency


def carbon_comparisons(it_mwh, process_heat_mwh, useful_heat_mwh, integrated_aux_mwh,
                       conventional_dry_aux_mwh, electricity_factor_g_per_kwh=43.0):
    ef_heat=gas_heat_lifecycle_tco2e_per_mwh_useful(); ef=electricity_factor_g_per_kwh/1000
    residual=max(process_heat_mwh-useful_heat_mwh,0)
    base=(it_mwh+conventional_dry_aux_mwh)*ef+process_heat_mwh*ef_heat
    integ=(it_mwh+integrated_aux_mwh)*ef+residual*ef_heat
    gas_base=(it_mwh+conventional_dry_aux_mwh)*.486+process_heat_mwh*ef_heat
    pv_integ=(it_mwh+integrated_aux_mwh)*.043+residual*ef_heat
    return {"same_supply_reduction_pct":100*(base-integ)/base,
            "same_supply_baseline_tCO2e":base,"same_supply_integrated_tCO2e":integ,
            "gas_to_pv_transition_reduction_pct":100*(gas_base-pv_integ)/gas_base}


def network_power_fraction(hour):
    lh=hour%24; dow=(hour//24)%7
    u=.62+.16*math.exp(-((lh-14)/5.0)**2)-.10*math.exp(-((lh-3)/3.2)**2)
    if dow>=5: u*=.96
    return float(np.clip(u,.45,.80))


def proportional_capped_allocation(weights,target,caps):
    a=np.zeros_like(caps,float); remaining=float(target); active=np.ones(len(caps),bool)
    w=np.maximum(np.asarray(weights,float),1e-12)
    for _ in range(20):
        if remaining<=1e-9 or not active.any(): break
        ww=w*active
        if ww.sum()<=0: ww=active.astype(float)
        proposal=remaining*ww/ww.sum(); room=caps-a; a+=np.minimum(proposal,room)
        remaining=target-a.sum(); active=(caps-a)>1e-9
    return a


def build_sites(gpu_multiplier=1.0,laundry_heat_multiplier=1.0,car_heat_multiplier=1.0,n_per_class=N_PER_CLASS):
    rows=[]; idx=0; total=len(GPU_COUNTS)*n_per_class; golden=math.pi*(3-math.sqrt(5))
    for cls,g0 in GPU_COUNTS.items():
        typ="laundry" if "laundromat" in cls else "car"
        mult=laundry_heat_multiplier if typ=="laundry" else car_heat_multiplier
        for i in range(n_per_class):
            # Preserve per-accelerator-equivalent thermal-sizing resolution. These values
            # are scientific sizing variables, not a claim that procurement occurs in
            # fractional or non-standard eight-GPU appliances.
            gpus=float(g0)*float(gpu_multiplier); rated=gpus*GPU_SYS_KW
            mean_heat=THROUGHPUT_PER_DAY[cls]*PROCESS_HEAT_KWH_PER_SERVICE[cls]*mult/24
            frac=(idx+.5)/total; radius=35*math.sqrt(frac); theta=idx*golden; idx+=1
            rows.append({"site":f"{cls} {i+1:02d}","facility_class":cls,"gpus":gpus,
                         "rated_it_kW":rated,"mean_heat_demand_kW":mean_heat,
                         "distance_to_core_straight_km":radius,
                         "distance_to_core_routed_km":radius*ROUTE_FACTOR,
                         "one_way_route_adjusted_us":radius*ROUTE_FACTOR*FIBRE_US_PER_KM,
                         "x_km":radius*math.cos(theta),"y_km":radius*math.sin(theta)})
    return pd.DataFrame(rows)


def pairwise_via_core_metrics(sites):
    vals=np.sort(sites.one_way_route_adjusted_us.to_numpy(float))
    if len(vals) < 2:
        one_way=float(vals[-1]) if len(vals) else 0.0
    else:
        one_way=float(vals[-1]+vals[-2])
    return one_way, 2.0*one_way


def run_scenario(wx, capture=.85,storage_hours=4,pvt_area_m2_per_kW=2.2,evap_cap_fraction=.20,
                 process_recovery_L_per_kWhth=4.0,power_mode="profile",heat_following=True,
                 gpu_multiplier=1.0,pump_frac_it=LIQUID_LOOP_PUMP_FRAC_IT,
                 laundry_heat_multiplier=1.0,car_heat_multiplier=1.0,n_per_class=N_PER_CLASS):
    sites=build_sites(gpu_multiplier,laundry_heat_multiplier,car_heat_multiplier,n_per_class=n_per_class)
    rated=sites.rated_it_kW.to_numpy(float); mean_heat=sites.mean_heat_demand_kW.to_numpy(float); N=len(sites)
    storage=np.zeros(N); storage_cap=mean_heat*storage_hours; area=rated*pvt_area_m2_per_kW
    shapes=np.array([LAUNDRY_SHAPE if "laundromat" in c else CAR_SHAPE for c in sites.facility_class])
    flux=rejection_flux(wx)
    sums={k:0.0 for k in ["it","captured","uncaptured","demand","useful","pvt","evap","captured_dry",
                           "unrejected","water","municipal","uncaptured_fan","captured_fan","evap_fan","pump"]}
    util=[]
    for h in range(HOURS):
        u=network_power_fraction(h) if power_mode=="profile" else float(power_mode.replace("flat","")); util.append(u)
        season=1+.05*math.cos(2*math.pi*((h/24)-15)/365.25)
        demand=mean_heat*shapes[:,h%24]*season; target=u*rated.sum()
        weights=(demand+.20*np.maximum(storage_cap-storage,0)+.03*mean_heat) if heat_following else rated
        it=proportional_capped_allocation(weights,target,rated); captured=capture*it; uncaptured=(1-capture)*it
        direct=np.minimum(captured,demand); rem=demand-direct; discharge=np.minimum(storage,rem); storage-=discharge
        surplus=np.maximum(captured-direct,0); charge=np.minimum(surplus,np.maximum(storage_cap-storage,0)); storage+=charge; surplus-=charge
        storage*=.998
        pvt_take=np.minimum(surplus,flux[h]*area/1000); surplus-=pvt_take
        evap_cap=evap_cap_fraction*rated if (wx["ghi"][h]>80 and wx["t_air"][h]>10) else np.zeros(N)
        evap=np.minimum(surplus,evap_cap); surplus-=evap
        water=1.6*evap; recovered=process_recovery_L_per_kWhth*demand; municipal=np.maximum(water-recovered,0)
        captured_dry=np.minimum(surplus,1.2*rated); surplus-=captured_dry
        for key,val in [("it",it.sum()),("captured",captured.sum()),("uncaptured",uncaptured.sum()),("demand",demand.sum()),
                        ("useful",(direct+discharge).sum()),("pvt",pvt_take.sum()),("evap",evap.sum()),
                        ("captured_dry",captured_dry.sum()),("unrejected",surplus.sum()),("water",water.sum()),
                        ("municipal",municipal.sum()),("uncaptured_fan",uncaptured.sum()/DRY_REJECTION_THERMAL_PER_ELECTRIC),
                        ("captured_fan",captured_dry.sum()/DRY_REJECTION_THERMAL_PER_ELECTRIC),
                        ("evap_fan",evap.sum()/DRY_REJECTION_THERMAL_PER_ELECTRIC),("pump",pump_frac_it*it.sum())]: sums[key]+=val
    it=sums["it"]/1000; demand=sums["demand"]/1000; useful=sums["useful"]/1000; captured=sums["captured"]/1000
    aux=(sums["uncaptured_fan"]+sums["captured_fan"]+sums["evap_fan"]+sums["pump"])/1000; conv_aux=it/DRY_REJECTION_THERMAL_PER_ELECTRIC
    pair_one_way,pair_roundtrip=pairwise_via_core_metrics(sites)
    total_gpus=float(sites.gpus.sum())
    total_gpus_out=int(round(total_gpus)) if abs(total_gpus-round(total_gpus)) < 1e-9 else total_gpus
    summary={"sites":N,"gpus":total_gpus_out,"rated_it_MW":rated.sum()/1000,"mean_it_power_fraction":np.mean(util),
             "annual_it_MWh":it,"process_heat_demand_MWh":demand,"useful_process_heat_MWh":useful,
             "captured_it_heat_MWh":captured,"captured_heat_utilisation_pct":100*useful/captured,
             "it_to_useful_heat_pct":100*useful/it,"heat_demand_coverage_pct":100*useful/demand,
             "uncaptured_it_heat_MWh":sums["uncaptured"]/1000,"pvt_rejection_MWh":sums["pvt"]/1000,
             "adiabatic_rejection_MWh":sums["evap"]/1000,"captured_dry_rejection_MWh":sums["captured_dry"]/1000,
             "unrejected_MWh":sums["unrejected"]/1000,"cooling_water_m3":sums["water"]/1000,
             "cooling_WUE_L_per_kWh_IT":sums["water"]/sums["it"],"municipal_cooling_water_m3":sums["municipal"]/1000,
             "annual_aux_MWh":aux,"auxiliary_pct_of_IT":100*aux/it,"conventional_dry_aux_MWh":conv_aux,
             "median_one_way_route_adjusted_us":sites.one_way_route_adjusted_us.median(),
             "p95_one_way_route_adjusted_us":sites.one_way_route_adjusted_us.quantile(.95),
             "max_one_way_route_adjusted_us":sites.one_way_route_adjusted_us.max(),
             "aggregate_installed_gpu_endpoint_Pbps":sites.gpus.sum()*GPU_ENDPOINT_GBPS/1e6,
             "max_pairwise_via_core_one_way_us":pair_one_way,
             "max_pairwise_via_core_roundtrip_us":pair_roundtrip}
    summary.update(carbon_comparisons(it,demand,useful,aux,conv_aux,43))
    return summary, sites


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--epw",help="Path to local EPW; otherwise downloads public CZ2025 LAX EPW")
    ap.add_argument("--out",default="."); args=ap.parse_args(); out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    wx=read_epw(args.epw); central,sites=run_scenario(wx)
    pd.DataFrame([central]).to_csv(out/"central_summary.csv",index=False); sites.to_csv(out/"network_sites.csv",index=False)
    scenarios=[("profile_heat_following",{}),("profile_no_scheduler",{"heat_following":False}),
               ("flat_0.649",{"power_mode":"flat0.649","heat_following":False}),
               ("flat_0.80",{"power_mode":"flat0.80"}),("flat_0.80_no_scheduler",{"power_mode":"flat0.80","heat_following":False}),
               ("flat_0.90",{"power_mode":"flat0.90"}),("flat_0.90_no_scheduler",{"power_mode":"flat0.90","heat_following":False}),
               ("flat_0.95",{"power_mode":"flat0.95"}),("flat_0.95_no_scheduler",{"power_mode":"flat0.95","heat_following":False}),
               ("dry_only",{"evap_cap_fraction":0}),
               ("gpu_multiplier_0.75",{"gpu_multiplier":.75}),("gpu_multiplier_1.25",{"gpu_multiplier":1.25}),
               ("gpu_multiplier_1.50",{"gpu_multiplier":1.5}),("capture_0.80",{"capture":.80}),("capture_0.95",{"capture":.95})]
    rows=[]
    for name,kw in scenarios:
        s,_=run_scenario(wx,**kw); rows.append({"scenario":name,**s})
    sens_df=pd.DataFrame(rows)
    sens_cols=["scenario","gpus","rated_it_MW","mean_it_power_fraction","it_to_useful_heat_pct",
               "heat_demand_coverage_pct","cooling_water_m3","same_supply_reduction_pct","captured_dry_rejection_MWh"]
    sens_df=sens_df[sens_cols].rename(columns={"same_supply_reduction_pct":"same_supply_reduction_pct_at_43g"})
    sens_df.to_csv(out/"sensitivity.csv",index=False)
    hrows=[]
    for m in [.5,.625,.75,1,1.25]:
        s,_=run_scenario(wx,laundry_heat_multiplier=m); hrows.append({"laundry_heat_multiplier":m,"laundry_kWhth_per_load":4.96*m,**s})
    pd.DataFrame(hrows).to_csv(out/"laundry_heat_sensitivity.csv",index=False)
    crows=[]
    for ef in [13,43,100,250,486]:
        c=carbon_comparisons(central["annual_it_MWh"],central["process_heat_demand_MWh"],central["useful_process_heat_MWh"],
                             central["annual_aux_MWh"],central["conventional_dry_aux_MWh"],ef)
        crows.append({"electricity_gCO2e_kWh":ef,**c})
    pd.DataFrame(crows).to_csv(out/"carbon_sensitivity.csv",index=False)
    pd.DataFrame([{"source":"California CZ2025 DEER weather files","station":"Los Angeles International Airport",
                   "WMO":722950,"years_basis":"TMY based on 2000-2020",
                   "mean_drybulb_C":wx["t_air"].mean(),"min_drybulb_C":wx["t_air"].min(),"max_drybulb_C":wx["t_air"].max(),
                   "mean_RH_pct":wx["rh"].mean()*100,"mean_wind_m_s":wx["wind"].mean(),"mean_GHI_W_m2":wx["ghi"].mean()}]).to_csv(out/"weather_source.csv",index=False)
    print(pd.Series(central).to_string())

if __name__=="__main__": main()
