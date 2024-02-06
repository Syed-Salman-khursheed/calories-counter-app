import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  TouchableOpacity,
  Dimensions
} from "react-native";
import { Button, Dialog, Portal } from 'react-native-paper';
import { InputField } from "./Register";
import DropDownPicker from 'react-native-dropdown-picker';
import * as yup from 'yup';
import { Formik } from 'formik';
import React, {useContext, useState} from 'react';
import {AuthAction} from '../Context/AuthContext';

const windowWidth = Dimensions.get('window').width;

const RegistrationValidationSchema = yup.object().shape({
  activity: yup.string().required('Activity level is required'),
  gender: yup.string().required('Gender is required'),
  goal: yup.string().required('Goal is required'),
  height: yup.number().required('Height is required'),
  currentWeight: yup.number().required('Current weight is required'),
  kgToGainLose: yup.number().required('Goal weight is required'),
  months: yup.number().required('Number of months is required'),
  age: yup.number().required('Age is required'),
});

export default function Registration({ navigation }) {

  const {adduserInfo} = useContext(AuthAction);
  const [visible, setVisible] = useState(false);
  const [data, setData] = useState(null);
  const showDialog = () => setVisible(true);
  const hideDialog = () => {
    adduserInfo(data,calories,navigation)
    setVisible(false)
  }

  const [calories, setCalories] = useState(null);
  const [open, setOpen] = useState(false);
  const [openGender, setOpenGender] = useState(false);
  const [openGoal, setOpenGoal] = useState(false);
  const [activity, setActivity] = useState(null);
  const [gender, setGender] = useState(null);
  const [goal, setGoal] = useState(null);
  const Activitylist = [
    { label: 'Sedentary', value: '1.2' },
    { label: 'lightly Active', value: '1.375' },
    { label: 'Moderately Active', value: '1.55' },
    { label: 'Very Active', value: '1.725' },
    { label: 'Extremely Active', value: '1.9' },
  ]
  const goallist = [
    { label: 'Gain Weight', value: 'gain' },
    { label: 'Lose Weight', value: 'lose' }
  ];
  const genderlist = [
    { label: 'Male', value: 'male' },
    { label: 'Female', value: 'female' }
  ];

  function handleCalories(value) {
    setData(value)
    console.log(value);
    const {
      activity,
      gender,
      goal,
      age,
      currentWeight,
      kgToGainLose,
      height,
      months,
    } = value;

    let bmr;
    if (parseFloat(currentWeight) <= 40 && goal === 'lose') {
      alert('Your weight is too low. Consult a healthcare professional.');
      return;
    }
    else if(parseFloat(currentWeight) >= 120 && goal === 'gain') {
      alert('Your weight is too high. Consult a healthcare professional.');
      return;
    }
    else if (gender === 'male') {
      bmr = 88.362 + 13.397 * currentWeight + 4.799 * height - 5.677 * age;
    } else {
      bmr = 447.593 + 9.247 * currentWeight + 3.098 * height - 4.33 * age;
    }

    const tdee = bmr * activity;
    const dailyCalories = Math.round(tdee);
    const goalMultiplier = goal === 'lose' ? -1 : 1;
    const totalGoalCalories = goalMultiplier * 3500 * months;
    const adjustedDailyCalories = dailyCalories + Math.round(totalGoalCalories / (months * 30));
    setCalories(adjustedDailyCalories)
    showDialog()
  }

  return (
    <ScrollView shhowsVerticalScrollIndicator={false}>
      <Formik
        validationSchema={RegistrationValidationSchema}
        initialValues={{
          activity: '',
          gender: '',
          goal: '',
          height: '',
          currentWeight: '',
          kgToGainLose: '',
          months: '',
          age: '',
        }}
        onSubmit={values => handleCalories(values)}
      >
        {({
          handleChange,
          handleBlur,
          handleSubmit,
          values,
          errors, 
          isValid, 
          setFieldValue, 
        }) => (
          <View style={styles.container}>
            {/* <KeyboardAvoidingView> */}
            <Text style={styles.topheading}>Set up Your profile</Text>
            <View>
           <DropDownPicker
              open={open}
              items={Activitylist}
              setOpen={() => setOpen(!open)}
              value={activity}
              setValue={(e,) => {
              setFieldValue('activity', e());
              setActivity(e());
              }}
              placeholder="Select Your Activity Level"
              disableBorderRadius={false}
              style={{ borderColor: '#01714b' }}
              placeholderStyle={{ color: '#01714b', fontWeight: 'bold' }}
              containerStyle={{ width: windowWidth * 0.8, marginBottom: 7 }}
              zIndex={5000}
            />
            <Text style={styles.error}>{errors.activity}</Text>
           </View>

          <View>
          <DropDownPicker
              open={openGender}
              items={genderlist}
              setOpen={() => setOpenGender(!openGender)}
              value={gender}
                setValue={(e,) => {
                  setFieldValue('gender', e());
                  setGender(e());
                }}
              placeholder="Select Your Gender"
              disableBorderRadius={false}
              style={{ borderColor: '#01714b' }}
              placeholderStyle={{ color: '#01714b', fontWeight: 'bold' }}
              containerStyle={{ width: windowWidth * 0.8, marginBottom: 7 }}
              zIndex={4000}
            />
            <Text style={styles.error}>{errors.gender}</Text>
          </View>

            <View>
            <DropDownPicker
              open={openGoal}
              items={goallist}
              setOpen={() => setOpenGoal(!openGoal)}
              value={goal}
              setValue={(e,) => {
                setFieldValue('goal', e());
                setGoal(e());
              }}
              placeholder="Select Your Goal"
              disableBorderRadius={false}
              style={{ borderColor: '#01714b' }}
              placeholderStyle={{ color: '#01714b', fontWeight: 'bold' }}
              containerStyle={{ width: windowWidth * 0.8, marginBottom: 7 }}
              zIndex={3000}
            />
             <Text style={styles.error}>{errors.goal}</Text>
            </View>

            <View style={styles.inputContainer}>
              <Text style={styles.heading}>What is your curent Height in cm?</Text>
              <InputField
                keyboardType="numeric"
                textContentType="givenName"
                placeholder="Enter Your Height in ft"
                placeholderTextColor="grey"
                onChangeText={handleChange('height')}
                
                name='height'
              />
              {errors.height &&
                <Text style={styles.error}>{errors.height}</Text>
              }
            </View>
            <View style={styles.inputContainer}>
              <Text style={styles.heading}>What is your curent weight in Kg?</Text>
              <InputField
                keyboardType="numeric"
                textContentType="givenName"
                placeholder="Enter Your curent weight in Kg"

                placeholderTextColor="grey"
                onChangeText={handleChange('currentWeight')}
                name='currentWeight'
              />
              {errors.currentWeight &&
                <Text style={styles.error}>{errors.currentWeight}</Text>
              }
            </View>
            <View style={styles.inputContainer}>

              <Text style={styles.heading}>
                How much weight would you like to Lose/Gain in kg?
              </Text>
              <InputField
                keyboardType="numeric"
                textContentType="givenName"
                placeholder="Enter weight in kg"
                placeholderTextColor="grey"
                onChangeText={handleChange('kgToGainLose')}
                name='kgToGainLose'
               />
               {errors.kgToGainLose &&
          <Text style={styles.error}>{errors.kgToGainLose}</Text>
        }
            </View>
            <View style={styles.inputContainer}>

              <Text style={styles.heading}>
                How much No. of month?

              </Text>
              <InputField
                keyboardType="numeric"
                textContentType="givenName"
                placeholder="Number of Months"
                placeholderTextColor="grey"
                onChangeText={handleChange('months')}
                name='months'
               />
               {errors.months &&
          <Text style={styles.error}>{errors.months}</Text>
        }
            </View>
            <View style={styles.inputContainer}>
              <Text style={styles.heading}>Enter Your Age</Text>
              <InputField
                keyboardType="numeric"
                textContentType="givenName"
                placeholder="Enter Your Age"
                placeholderTextColor="grey"
                onChangeText={handleChange('age')}
                name='age'
               />
               {errors.age &&
          <Text style={styles.error}>{errors.age}</Text>
        }
            </View>
            <TouchableOpacity style={styles.Btn} onPress={handleSubmit}>
              <Text style={styles.btntext}>Submit</Text>
            </TouchableOpacity>
            {/* </KeyboardAvoidingView> */}
          </View>
        )}
      </Formik>
      <View>
        <Portal>
          <Dialog visible={visible} onDismiss={hideDialog} style={{backgroundColor:'white',color:'#555'}}>
            <Dialog.Title>Calculated Calories</Dialog.Title>
            <Dialog.Content>
              <Text variant="bodyMedium" style={{color:'#555'}}>{`Your Per day daily calories required is ${calories}`}</Text>
            </Dialog.Content>
            <Dialog.Actions>
              <Button onPress={hideDialog} style={{color:'#01714b'}}>Done</Button>
            </Dialog.Actions>
          </Dialog>
        </Portal>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    width: windowWidth,
    backgroundColor: '#ffff',
    alignItems: "center",
    justifyContent: "center",
  },

  topheading: {
    fontSize: 32,
    padding: 20,
    marginTop: 5,
    marginBottom: 0,
    textAlign: "center",
    fontWeight: "bold",
    color: "#01714b",
  },
  inputContainer: {
    marginBottom: 7,
  },
  heading: {
    paddingLeft: 5,
    paddingBottom: 5,
    fontSize: 12,
    color: '#01714b'
  },
  Btn: {
    color: '#ffffff',
    backgroundColor: "#01714b",
    width: windowWidth * 0.8,
    padding: 12,
    borderTopLeftRadius: 15,
    borderTopRightRadius: 15,
    borderBottomLeftRadius: 15,
    borderBottomRightRadius: 15,
    marginTop: 10,
    marginBottom: 20,
    elevation: 2,
  },
  btntext: {
    color: "#fff",
    fontSize: 20,
    fontWeight: "bold",
    textAlign: "center",
  },
  error:{
    fontSize:11,
    color:'red',
    alignSelf:'flex-start',
    marginBottom:8,
    marginLeft:6,
  }
});
