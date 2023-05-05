# ----------------------------------------------------- clean up titanic -----------------------------------------------
require(dplyr)
require(readr)
require(magrittr)
`%>%` <- magrittr::`%>%`
#-----------------------------------------------------------------------------------------------------------------------
# funcitons:
#-----------------------------------------------------------------------------------------------------------------------
extract_prefix <- function(name, sex) {
  purrr::map2_chr(name, sex, ~{
    #prefix <- stringr::str_extract_all(., "(Mme)|(Mlle. )|([mM]iss)|([Mm]rs)|([Mm]r)|([Mm]aster)|(Dr)|(Rev)|(Col)|(Major)|(Capt. )|(Don. )|(Sir)|(Lady)|(Countess. )|(Jonkheer. )")

    # if Nob or prof, rare
    prof <- stringr::str_detect(.x, "(Dr)|(Rev)|(Col)|(Major)|(Capt. )")
    nob <- stringr::str_detect(.x, "(Don. )|(Dona.)|(Sir)|(Lady)|(Countess. )|(Jonkheer. )")
    if(nob | prof) { return("rare") }
    
    # if married female, Mrs 
    # (inc professional females - these are likely to be assuming their husbands title because sexism and all that, right?)
    if(stringr::str_detect(.x, "(Mlle. )|([Mm]rs)") | (prof & .y == 1) | (stringr::str_detect(.x, "Mr") & .y == 1)) {return("Mrs")}
    
    # If professional, Prof
    if(prof) {return("prof")}
      
    # if unmarried female, Miss
    if(stringr::str_detect(.x, "(Mme)|([mM]iss)")) {return("Miss")}
    
    if(stringr::str_detect(.x, "Mr")) {return("Mr")}
    if(stringr::str_detect(.x, "Master")) {return("Master")}
    if(stringr::str_detect(.x, "[Mm]s")) {return("Ms")}
  })
}

#extract_prefix(titanic$name, titanic$sex)
# Mr, Mrs, Miss, Master, Prof, Nob


#-----------------------------------------------------------------------------------------------------------------------
# read in data and clean:
#-----------------------------------------------------------------------------------------------------------------------
titanic_train <- readr::read_csv("titanic/train.csv") %>% dplyr::mutate(set = "train")
titanic_test <- readr::read_csv("titanic/test.csv") %>% dplyr::mutate(set = "test")

titanic <- dplyr::bind_rows(titanic_train, titanic_test) %>%
  dplyr::transmute(
    PassengerId = PassengerId,
    name = Name,
    ticket = Ticket,
    survived = as.integer( Survived ),
    class = as.integer( Pclass ),
    sex = ifelse(Sex == "male", 0L, 1L),
    age = Age,
    sib_sp = as.integer( SibSp ),
    parch = as.integer( Parch ),
    fare = Fare,
    deck = stringr::str_extract(Cabin, "^[a-zA-Z]"),
    n_cabins = purrr::map_dbl(Cabin, ~{if(is.na(.)) {return(NA)} else {length(stringr::str_split(., " "))}}),
    embarked = Embarked,
    set = set
  )


#-----------------------------------------------------------------------------------------------------------------------
# look for relationships to fill-in missing values:
#-----------------------------------------------------------------------------------------------------------------------

# problem: --------------------------------------------------------------------
# age might be important but there are 86 NAs in age for the test data 

# observations: ---------------------------------------------------------------
# - class has some relationship to age (the rich are older) but fare does not.
# - sib_sp and parch (i.e. family size) are both negatively related to age.
# - lower decks with lower classes appear to be younger but lots of missing data (maybe means no cabin assigned?)
# - there are no NAs in class sib_sp or parch

# solution: ------------------------------------------------------------------
# - for the test data set estimate missing ages from class, sib_sp and parch
# - for the training data remove missing rows (?)

# problem: --------------------------------------------------------------------
# - fare might be important but there there is a single missing fare value in test

# observations: ---------------------------------------------------------------
# - the missing value is class 3, no family group, 60.5yrs old and unknown deck
# - there is a strong relationship between class and fare, especially for upper class
# - higher fairs for more central decks (but widely dispersed unknowns)
# - some relationship between sib_sp and parch, especially for lowest class
# - some relationship between embarkation port and fare, S is most expensive in lower class
# - little relationship between age or sex and fare

# solution: -------------------------------------------------------------------
# - Estimate the missing fare from sib_sp, parch and embarkation port
# - ignore class 2 and 3 in the estimation


#-----------------------------------------------------------------------------------------------------------------------
# fill missing values
#-----------------------------------------------------------------------------------------------------------------------

fare_mod <- lm(fare ~ sib_sp + parch, data = titanic, subset = class == 3)
age_mod <- lm(age ~ factor(class) + sib_sp + parch, data = titanic) 

titanic <- titanic %>%
  dplyr::mutate(
    fare = ifelse(is.na(fare), 8.4549, fare),
    age = purrr::pmap_dbl(list(age, class, parch, sib_sp), function(age, class, parch, sib_sp){
      if(!is.na(age)){
        return(age)
      } else {
        predict(age_mod, list("class" = class, "parch" = parch, "sib_sp" = sib_sp)) %>% pmax(.,0)
        }
    })
  )

#-----------------------------------------------------------------------------------------------------------------------
# feature engineering
#-----------------------------------------------------------------------------------------------------------------------

titanic <- titanic %>%
  dplyr::mutate(
    prefix = extract_prefix(name, sex),
    family_size = sib_sp + parch,
    #married = purrr::map_int(prefix, ~{stringr::str_detect(., "(Mr)|(Mrs)|(Mme)") %>% any()}),
    #with_spouse = as.integer( married & sib_sp > 0 ),
    #with_dependants = as.integer( married & parch > 0 ),
    #socialite = purrr::map_int(prefix, ~{stringr::str_detect(., "(Rev)|(Dr)|(Col)|(Major)|(Sir)|(Lady)") %>% any()}),
    fare_bracket = purrr::map_chr(fare, function(fare) {
      if(fare < 25) return("Fare_lowest")
      if(fare < 50) return("Fare_low")
      if(fare < 100) return("Fare_med")
      if(fare < 200) return("Fare_high")
      return("Fare_very_high")
    }),
    age_bracket = purrr::map_chr(age, function(age) {
      if(age < 12) return("child")
      if(age < 20) return("teenager")
      if(age < 40) return("adult")
      return("old")
    })
  ) %>%
  dplyr::select(PassengerId, survived, prefix, class, sex, age, fare, deck, embarked, family_size, fare_bracket, age_bracket, set)


#-----------------------------------------------------------------------------------------------------------------------
# possible feature engineering approaches:
#-----------------------------------------------------------------------------------------------------------------------

titanic %>%
  dplyr::mutate(fare = round(titanic$fare) ) %>%
  dplyr::arrange(fare) %>%
  dplyr::group_by(fare) %>%
  dplyr::summarise(
    n = dplyr::n()
  ) %>%
  ggplot2::ggplot( ggplot2::aes( fare, n ) ) +
  ggplot2::geom_col() +
  ggplot2::scale_x_continuous(breaks = seq(0, 500, 25)) +
  ggplot2::coord_cartesian(xlim = c(0, 50))

# family size:
# parch + sib_sp

# age structure

# adult female with dependents: sex = 1 & age >= 20 & parch > 0
# adult male with dependants: sex = 0 & age >= 20 & parch > 0

# titles from names




#-----------------------------------------------------------------------------------------------------------------------
# read out:
#-----------------------------------------------------------------------------------------------------------------------
readr::write_csv(titanic, "titanic/titanic_clean.csv")

# 0.8518518518518519