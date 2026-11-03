zcui14=2.2;
Mcui14=[12.5,13.7];
dndndMcui14=[1.4,2.5];
Mcui140=[12.5,13.,13.5,14.,14.5,15.];
dndndMcui140=interp([1.1,1.5],[12.5,15.],Mcui140);
zvog14=0.;
Mvog14=[8.5,9.5,10.5,11.5,12.5,13.5,14.5];
dndndMvog14=[0.75,0.75,0.9,1.,0.9,0.8,0.9];
zSaw14=0.;
Msaw14=[9.5,10.5,11.5,12.5,13.5];
dndndMsaw14=[0.75,0.75,0.8,0.9,1.];


//what could go wrong with this though: 
//- currently we're using the Mfof=(1+(omegab-omegam)/omegam)*MfofDM !! while what we should be really using is MfofDM+Mstar(R200)+Mgas(R200). That could also be wrong given the stupid morphology of the fof groups. and we might have to use instead MDM=MDM(R200) instead of MfofDM... ppffffff
//- also, we're comparing our MF against another one which was obtained in different conditions (in particular force resolution), and not with the pure DM counterpart of coda....
//However, if we had that pure DM counterpart, it would be interesting to do studies such as:
//- MF differences at various z
//- MF differences in various environments.

//so what are the options?
//- drop the MF section, just say: we've made FoF halo catalogs. and tahts it.
//- lets hope Nicolas' computations get it right and then we can leave the lowest z MF and discuss its difference with the universal FoF. we know it differes for at least 3 possible reasons:
//  - baryon impact on MF and halo formation. Nice, tahts physical, plenty of literature to back us up.
//  - overlinking: yes but the watson fof has it as well so no...
//  - poor force resolution aggravated overlinking.
//  - Models without AGN are notorious for overestimating the number of bright gals but usually thats not because there are too many massive DM haloes, but just because they have too strong SFR. Cui14 is the only one in the literature we consider here to show what happens without AGN, and show that actually including the AGN is super important at high mass. Could it be that AGN also impact the hal mass function at high z? NOTE also that Cui2014 is the only paper to have Mhyd/MDM>1 at high mass, but also the only one to show no AGN sim.... what does that tell us...???
//  - // Velliscig 2014 has a SF + SN model which does not behave like Cui2014's SF + SN: Cui2014 for some reason is alone in that crazy corner: indeed, how would that even be possible? has to be overlinking?
//  thats still a reasonably interesting discussion...
// 



