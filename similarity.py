from jd_parsing import scrap_jd, jdParser
import numpy as np
from mongo import resume

link = "https://www.linkedin.com/jobs/view/3903269267"

def cosine_similarity(v1,v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))


def findBestMatch(job_link):
    jd = scrap_jd(job_link)
    if jd:
        parse_jd = jdParser(jd)
        jd_skill_vector = parse_jd["skill_embedding"]
        cursor = resume.find()
        for doc in cursor:
            dic = {}
            resume_skill_vector = doc["skill_embedding"]
            sim  = cosine_similarity(jd_skill_vector, resume_skill_vector)
            dic[doc["email"]] = sim
        
        best_candidate = sorted(dic.items(), key = lambda x:x[1], reverse = True)[:10]
        
        return best_candidate
print(findBestMatch(link))