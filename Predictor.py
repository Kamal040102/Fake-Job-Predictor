import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


class Predictor:
  def predictor(self, userInp):
    df = pandas.read_csv("./data.csv")

    # df.fillna(" ", inplace=True)

    # df['text'] = df['title'] + " " + df["location"] + " " + df["department"] + " " + df["company_profile"] + " " + df["description"] + " " + df["requirements"] + " " + df["benefits"] + " " + df["industry"] + " " + df["function"]

    # df.drop(["title", "location", "department", "company_profile", "description", "requirements", "benefits", "employment_type", "required_experience", "required_education", "industry", "function", "job_id", "salary_range", "telecommuting", "has_company_logo", "has_questions"], axis = 1, inplace=True)


    # df['text'] = df.text.apply(lambda x: x.lower())
    # stopwordsenglish = stopwords.words("english")
    # df['text'] = df.text.apply(lambda x : ' '.join(words for words in x.split() if words not in stopwordsenglish))

    # df.to_csv("data.csv", index=False)

    X_train, X_test, Y_train, Y_test = train_test_split(df['text'], df['fraudulent'], test_size=0.3)
    vect = TfidfVectorizer()
    vect.fit(X_train)
    X_train_dtm = vect.transform(X_train)
    # X_test_dtm = vect.transform(X_test)

    dt = SVC(kernel='linear', random_state=0)
    dt.fit(X_train_dtm, Y_train)
    # dt_prediction = dt.predict(X_test_dtm)

    userInp = [userInp]
    userInp_dtm = vect.transform(userInp)

    result = dt.predict(userInp_dtm)

    if result[0] == 0:
      answer = "It is a Real Job"
    else:
      answer = "It is a fake job"

    return answer

#  Non-Fraud:
# "medical coder us, tx, fort worth spotsource solutions llc global human capital management consulting firm headquartered miami, florida. founded january 2012, spotsource created fusion innovative service offerings meet increasing demand today’s economy. specialize talent acquisition, staffing, executive search services across various functions specific industries. global talent transfusion (gtt) services utilize best practice qualification standards deliver talent temporary, temporary-to-hire, permanent basis. health career transition (hct) subsidiary global talent transfusion offers placement services specifically growing healthcare arena. spotsource executive search (ses) consultants special breed talent evangelists understand advise streamline human resources process direction organization requires long term sustainability success. succession planning. leadership development programs. compensation analysis. recruitment process outsourcing. customized best suit needs business. understand demand cost-effective solutions organization. seeking potential career transition interested discussing current hiring trends open positions? vital career consulting (vcc) offers career transitional services catered specifically job applicant, including resume construction, social media optimization, interview coaching.address:4100 n powerline rd. ste z3pompano beach fl 33073office: #phone_c90b6ca89acd18d9e31ea3590b4ad76605721bc372025598fe9e1e60cf428551# job title: certified coder department: itemization review reports to: itemization review supervisorgeneral description:reverse code previously coded medical bills determine coding accuracy. duties responsibilities:receives claim processes based state rules regulationsdetermine validity compensability claim using proprietary programsmake recommendations referring officecommunicate claim status referring officeread comprehend medical aapc reportsadhere client carrier guidelines participate claims review neededassists claims professionals complex problematic claims necessaryadditional duties/responsibilities assignedcomply safety rules/regulations, conjunction injury illness prevention program (“iipp”), well as, maintain hipaa compliancequalifications:high school diploma, equivalentcurrent aapc certification (which must maintained throughout employment current active status)certification cpc aapc 5 years (w/ surgical office experience)current recent orthopedic billing/coding experiencee/m coding/down-coding experienceencoderpro software experiencetexas workers compensation experience preferredpain management/anesthesia/general surgery coding experience preferredability learn rapidly develop knowledge understanding claims practicestrong organizational skillsability meet exceed performance competencieseffective professional communication skillsability handle stressful situations, use critical strategic thinkingdemonstrated outstanding leadership, problem solving, analytical skillsability think work independently, working overall team environmentproficient microsoft office suite production expectations:meet claim review quota 25 claims per daymaintain error rate 2% less benefits offered hospital & health care"

# Fraud:
# "class - cdl driver - doubles endorsed us, oh, cincinnati bradley contracting group corporation offers wide range b2b services plethora different small business entities. sectors many different industries. main objective harness contracts many different small businesses, corporations cincinnati tri-state area, southern ohio region.bradley contracting group corporation around almost 5 years work different cities various clients. we've worked louisville ky, charlotte nc, atlanta ga. november 2013 we've established transportation courier service. clients note fedex ground, hhgreg. result diligence dedication excellence we're also looking expand markets well obtain contracts lowes, best buy, city cincinnati, state ohio, many different schools districts schools well. we're excellent standing state kentucky registered foreign entity state ohio. we're temporarily stationed hamilton, ohio within transportation warehousing facility we've recently acquired. main focus establish reputation excellence, diligence, innovation. feel live motto, ""excellence excuses."", we'd love part team! deliver freight customer safe efficient manner, adhering company policies times, working team. high school diploma equivalentat least 1 year cdl experience past 5 yearsvalid class - commercial driver’s license interstate operationequivalent equipment experience (reefer, flatbed, drybed, etc)linehaul experience plus. determined. logistics supply chain distribution"
