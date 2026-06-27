import json
import datetime

current_date=datetime.datetime(2026,5,24)

service_based_companies={"tcs", "infosys", "wipro", "accenture", "cognizant", "mindtree", "hcl", "tech mahindra"}
IR_keywords={"nlp", "search", "retrieval", "ranking", "embedding", "vector", "llm", "transformers", "milvus", "pinecone", "faiss", "elasticsearch", "lora"}

def three_level_architecture(candidate_chunks):
    cleaned_candidates=[]

    for candidate in candidate_chunks:

        #LEVEL 1
        red_sig=candidate.get('redrob_signals',{})

        if red_sig.get('search_appearance_30d',0)>2000 and red_sig.get('saved_by_recruiters_30d',0)==0:
            continue

        assessments=red_sig.get('skill_assessment_scores',{})

        if 'NLP' in assessments and assessments['NLP']<40.0:
            continue
        if 'Fine-tuning LLMs' in assessments and assessments['Fine-tuning LLMs']<40.0:
            continue

        salary = red_sig.get('expected_salary_range_inr_lpa', {})

        if salary:
            salary_min = salary.get('min', 0)
            salary_max = salary.get('max', 0)

            if salary_min > salary_max:
                continue

        #LEVEL 2 HARD DISQUALIFICATION

        history=candidate.get('career_history',[])

        if history:

            companies=[str(job.get('company','')).lower().strip() for job in history]

            if all(any(service in comp for service in service_based_companies) for comp in companies):
                continue

            skills_raw=candidate.get('skills',[])

            skill_sets={str(s.get('name','').strip().lower()) for s in skills_raw if s}

            if not skill_sets.intersection(IR_keywords):
                continue

        #LEVEL 3 BEHAVIORAL MULTIPLIER CALCULATION

        # Ghosting score (1.Response multipler)
        response_multiplier=1.0
        response_rate=red_sig.get('recruiter_response_rate',0)
        
        if response_rate< 0.40:
            response_multiplier=0.6
        elif response_rate>=0.80:
            response_multiplier=1.2

        # Ghosting score (2.Dormance score)

        last_active=red_sig.get('last_active_date','')
        dormancy_multiplier=1.0

        if last_active:
            last_active_date=datetime.datetime.strptime(last_active,"%Y-%m-%d")

            days_inactive=(current_date-last_active_date).days

            if days_inactive > 180:
                dormancy_multiplier=0.5

            else:
                dormancy_multiplier=1.0
        
        #FINAL GHOSTING SCORE
        ghosting_score=response_multiplier*dormancy_multiplier

        #Experience multiplier

        total_days=0

        for job in history:
            start_str=job.get('start_date','')
            end_str=job.get('end_date','')

            if start_str:
                start_dt=datetime.datetime.strptime(start_str,"%Y-%m-%d")

                if end_str:
                    end_dt=datetime.datetime.strptime(end_str,"%Y-%m-%d")
                else:
                    end_dt=current_date

                total_days+=(end_dt-start_dt).days
        
        total_years=total_days/365.25

        experience_score=1.0
        if total_years < 5.0:
            experience_score=0.6
        elif 5.0 <= total_years <= 9.0:
            experience_score=1.2
        elif total_years > 9.0:
            experience_score=0.8

        #Github score
        github_multiplier=1.0

        github_score=red_sig.get('github_activity_score',-1)

        if github_score!=-1:
            github_multiplier=1.3

        #Location score
        location_multiplier=1.0
        prof=candidate.get('profile',{})
        cand_location=prof.get('location','').lower().strip()
        cand_country=prof.get('country','').lower().strip()

        is_willing_to_relocate=red_sig.get('willing_to_relocate',False)

        tier1_cities=["mumbai", "delhi", "gurgaon", "noida", "bengaluru", "chennai", "hyderabad", "kolkata", "pune", "ahmedabad"]

        if "pune" in cand_location or "noida" in cand_location:
            location_multiplier = 1.5

        elif ("hyderabad" in cand_location or 
              "gurgaon" in cand_location or 
              "mumbai" in cand_location or 
              "delhi" in cand_location):
            location_multiplier = 1.3

        elif any(city in cand_location for city in tier1_cities) and is_willing_to_relocate:
            location_multiplier = 1.2

        elif cand_country == "india" and is_willing_to_relocate:
            location_multiplier = 1.1

        elif cand_country=='india' and not is_willing_to_relocate:
            location_multiplier=1.0

        elif cand_country!='india' and is_willing_to_relocate:
            location_multiplier=0.9

        else:
            location_multiplier = 0.8

        #Title chaser multiplier
        title_chaser_multiplier=1.0
        career_hist=candidate.get('career_history',[])
        total_months=0

        if len(career_hist) >1:
            for job in career_hist:
                total_months+=job.get('duration_months',0)
            
            average_working=total_months/len(career_hist)

            if average_working <18:
                title_chaser_multiplier=0.7
        
        offer_acceptance_multiplier=1.0
        if red_sig.get('offer_acceptance_rate', -1) < 0 or not red_sig.get('verified_email', True) or not red_sig.get('verified_phone', True):
            offer_acceptance_multiplier=0.7

        profile = candidate.get('profile', {})
        current_title = str(profile.get('current_title', '')).lower()

        core_technical_keywords = ['machine learning', 'ml', 'ai', 'artificial intelligence', 'data scientist', 'backend', 'nlp', 'search', 'retrieval', 'software engineer','ranking','embeddings','llm','llms','fine tuning','tuning']
        honeypot_titles = ['','project','project manager','project management','java','cloud','full stack','devops','marketing', 'support', 'accountant', 'mechanical', 'operations', 'sales', 'hr', 'recruiter', 'customer support', 'customer', 'civil','.net','graphic','design','designer', 'analyst', 'business']

        title_alignment_multiplier = 1.2

        if any(honeypot in current_title for honeypot in honeypot_titles):
            if not any(core in current_title for core in core_technical_keywords):
                title_alignment_multiplier = 0.1
        
        #Behavioral Mutiplier FINAL calculation
        behavioral_multiplier=ghosting_score*experience_score*github_multiplier*location_multiplier*title_chaser_multiplier*offer_acceptance_multiplier*title_alignment_multiplier
            
        candidate['behavioral_multiplier']=round(behavioral_multiplier,4)
        cleaned_candidates.append(candidate)

    return cleaned_candidates