import {
  StyleSheet,
  Text,
  View,
  Image,
  ScrollView,
  TouchableOpacity,Dimensions
} from "react-native";
import * as yup from 'yup'
import { Formik } from 'formik'
import {ActivityIndicator} from 'react-native-paper';
import { InputField } from "./Register";
import React, {useContext, useState, useEffect} from 'react';
import {AuthAction} from '../Context/AuthContext';
import AsyncStorage from "@react-native-async-storage/async-storage";


const windowWidth = Dimensions.get('window').width;
const windowHeight = Dimensions.get('window').height;

const loginValidationSchema = yup.object().shape({
  email: yup
    .string()
    .email("Please enter valid email")
    .required('Email Address is Required'),
  password: yup
    .string()
    .min(8, ({ min }) => `Password must be at least ${min} characters`)
    .required('Password is required'),
})

export default function Login({ navigation }) {

  const {onSignIn} = useContext(AuthAction);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(value) {
    setLoading(true)
    onSignIn(value.email, value.password, navigation);
    setLoading(false)
  }

  useEffect(() => {
    registerSplash();
  });

  const registerSplash = async () => {
    const getUser = await AsyncStorage.getItem('User');
      if (getUser !== null) {
        navigation.navigate('Home');
      }
  };

  return (
    <View style={styles.container}>
      {/* <ScrollView s/howsVerticalScrollIndicator={false}> */}

        <Text style={styles.heading}>LOGIN</Text>

        <Formik
   validationSchema={loginValidationSchema}
   initialValues={{ email: '', password: '' }}
   onSubmit={values => handleSubmit(values)}
 >
   {({
     handleChange,
     handleBlur,
     handleSubmit,
     values,
     errors,
     isValid,
   }) => (
     <>
        <View>

        <InputField
          keyboardType="email-address"
          textContentType="emailAddress"
          placeholder="Enter Your Email"
          placeholderTextColor="grey"
          onChangeText={handleChange('email')}
          name='email'
          value={values.email}
        />
        {errors.email &&
         <Text style={styles.error}>{errors.email}</Text>
       }
        </View>

       <View>

       <InputField
          secureTextEntry={true}
          name="password"
          keyboardType="password"
          textContentType="password"
          placeholder="Enter Your Password"
          placeholderTextColor="grey"
          onChangeText={handleChange('password')}
          value={values.password}
          />
        {errors.password &&
         <Text style={styles.error}>{errors.password}</Text>
       }
       </View>

        <TouchableOpacity style={styles.Btn} activeOpacity={0.5}
        onPress={handleSubmit}>
            <Text style={styles.btntext}> {loading ? (
              <ActivityIndicator animating={true} color="#fff" />
            ) : 'LOG IN'}</Text>
        </TouchableOpacity>
        </>
   )}
 </Formik>
     
        <View style={styles.auth}>
        <TouchableOpacity 
        //  onPress={() => navigation.navigate("Signup")}
         >
            <Text
              style={styles.forget}
            >Forget Password?</Text>
          </TouchableOpacity>
          <TouchableOpacity 
         onPress={() => navigation.navigate("Register")}
         >
            <Text
              style={styles.signup}
            >Signup</Text>
          </TouchableOpacity>
        
        </View>
      {/* </ScrollView> */}
    </View>
  );
}
const styles = StyleSheet.create({
  loading: {
    flex: 1,
    backgroundColor: '#ffff',
    alignItems: "center",
    justifyContent: "center",
  },
  container: {
    width:windowWidth,
    flex: 1,
    backgroundColor: '#fafafa',
    justifyContent:'center',
    alignItems:'center',
  },
  heading: {
    // color: "#ff8b3d",
    color: "#01714b",
    fontSize: 46,
    fontWeight:'bold',
    marginBottom:40,
    // marginTop: 150,
    // marginBottom: 50,
    textAlign: "center",
  },
  loading: {
    width: 300,
    height: 200,
  },
  Btn: {
    color: '#fafafa',
    backgroundColor: "#01714b",
    width:windowWidth*0.8,
    padding: 12,
    borderTopLeftRadius: 15,
    borderTopRightRadius: 15,
    borderBottomLeftRadius: 15,
    borderBottomRightRadius: 15,
    marginTop: 20,
    elevation:2,

  },
  btntext: {
    color: '#fafafa',
    fontSize: 20,
    fontWeight:'bold',
    textAlign: 'center',
  },
  auth:{
    width:windowWidth*0.8,
    marginTop:20,
    padding:10,
    flexDirection: 'row',
    alignItems:'center',
    justifyContent:'space-between'
  },
  forget:{
    color: "#555",
    // marginTop: 35,
    // marginBottom: 10,
    // alignSelf:'center',
    fontWeight:"bold", 
  },
  signup:{
    color: "#01714b",
    // marginTop: 35,
    // marginBottom: 10,
    // alignSelf:'center',
    fontWeight:"bold",
    
  },
  error:{
    fontSize:11,
    color:'red',
    alignSelf:'flex-start',
    marginBottom:8,
    marginLeft:6,
  }
});
