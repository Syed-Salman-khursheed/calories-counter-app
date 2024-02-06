
import {
  StyleSheet, StatusBar, Text, View,
  TouchableOpacity, TextInput, ScrollView,
  KeyboardAvoidingView, Dimensions
} from 'react-native';
import * as yup from 'yup'
import { Formik } from 'formik'
import {ActivityIndicator} from 'react-native-paper';

import React, {useContext, useState} from 'react';
import {AuthAction} from '../Context/AuthContext';

const windowWidth = Dimensions.get('window').width;
const windowHeight = Dimensions.get('window').height;

const SignUpValidationSchema = yup.object().shape({
  firstName: yup.string().required('First name is Required'),
  lastName: yup.string().required('Last name is Required'),
  email: yup
    .string()
    .email("Please enter valid email")
    .required('Email Address is Required'),
  password: yup
    .string()
    .min(8, ({ min }) => `Password must be at least ${min} characters`)
    .required('Password is required'),
})

export default function Register({ navigation }) {
  const [loading, setLoading] = useState(false)
  const {onSignUp} = useContext(AuthAction);


  function handleSubmit(value) {
    setLoading(true)
    console.log('user---->>', value)
    onSignUp(value,navigation)   
  }

  return (
    <KeyboardAvoidingView style={styles.container}>
      <StatusBar style="light" />
      <ScrollView>
        <View style={styles.container}>

          <View style={styles.top}>
            <Text style={styles.heading} >Sign Up</Text>

          </View>
          <View style={styles.bottom}>

            <Formik
              validationSchema={SignUpValidationSchema}
              initialValues={{ firstName: '', lastName: '', email: '', password: '' }}
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
                    <InputField keyboardType='default'
                      textContentType='givenName'
                      placeholder="Enter Your First Name"
                      placeholderTextColor='grey'
                      onChangeText={handleChange('firstName')}
                      onBlur={handleBlur('firstName')}
                      value={values.firstName}
                      name='firstName'
                    />
                    {errors.firstName &&
                      <Text style={styles.error}>{errors.firstName}</Text>
                    }
                  </View>
                  <View>
                    <InputField keyboardType='default'
                      textContentType='givenName'
                      placeholder="Enter Your Last Name"
                      placeholderTextColor='grey'
                      onChangeText={handleChange('lastName')}
                      onBlur={handleBlur('lastName')}
                      name='lasstName'
                      value={values.lastName}
                    />
                    {errors.lastName &&
                      <Text style={styles.error}>{errors.lastName}</Text>
                    }
                  </View>

                  <View>
                    <InputField keyboardType='email-address'
                      textContentType='emailAddress'
                      placeholder="Enter Your Email"
                      placeholderTextColor='grey'
                      onChangeText={handleChange('email')}
                      onBlur={handleBlur('email')}
                      value={values.email}
                      name='email'
                    />
                    {errors.email &&
                      <Text style={styles.error}>{errors.email}</Text>
                    }
                  </View>

                  <View>
                    <InputField secureTextEntry={true}
                      keyboardType='password'
                      textContentType='password'
                      placeholder="Enter Your Password"
                      placeholderTextColor='grey'
                      onChangeText={handleChange('password')}
                      onBlur={handleBlur('password')}
                      value={values.password}
                      name='password'
                    />
                    {errors.password &&
                      <Text style={styles.error}>{errors.password}</Text>
                    }
                  </View>

                  <TouchableOpacity style={styles.Btn}
                    onPress={handleSubmit}>
                    <Text style={styles.btntext}> {loading ? (
                      <ActivityIndicator animating={true} color="#fff" />
                    ) : 'SIGN UP'}</Text>
                  </TouchableOpacity>


                </>
              )}
            </Formik>
            <TouchableOpacity
              activeOpacity={0.5}
              onPress={() => navigation.navigate("Login")}
              style={{ flexDirection: 'row', marginTop: 30, }}
            >
              <Text
                style={styles.already}
              >Already have an account yet? </Text>
              <Text
                style={styles.signin}
              >Sign In</Text>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>

  );
}
export function InputField({ keyboardType, placeholder, placeholderTextColor, onChangeText, textContentType, secureTextEntry }) {

  return (
    <TextInput style={styles.input}
      // autoFocus
      keyboardType={keyboardType}
      textContentType={textContentType}
      placeholder={placeholder}
      placeholderTextColor={placeholderTextColor}
      onChangeText={onChangeText}
      secureTextEntry={secureTextEntry} />
  )
}


const styles = StyleSheet.create({
  loading: {
    flex: 1,
    backgroundColor: '#fafafa',
    alignItems: "center",
    justifyContent: "center",
  },
  container: {
    width: windowWidth,
    height: windowHeight,
    flex: 1,
    backgroundColor: "#01714b",
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  },
  top: {
    flex: 1,
    justifyContent: 'center'
  },
  bottom: {
    flex: 2,
    backgroundColor: '#fafafa',
    width: windowWidth,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 20,
    // shadowColor:'red',.
    borderTopLeftRadius: 15,
    borderTopRightRadius: 15,
  },
  Logo: {
    marginTop: 50,
    width: 350,
    height: 130,
  },
  heading: {
    color: "#fafafa",
    fontSize: 48,
    fontWeight: 'bold',
    // alignItems:'center',
    // justifyContent:'center'
    // marginTop: 100,
    // marginBottom: 20,
  },
  input: {
    width: windowWidth * 0.8,
    height: 50,
    marginBottom: 7,
    paddingLeft: 15,
    borderRadius: 10,
    backgroundColor: '#e6f1ed',
    color: '#8e8e8e'
  },
  Btn: {
    backgroundColor: '#01714b',
    color: "#fafaf",
    width: windowWidth * 0.8,
    padding: 15,
    borderRadius: 10,
    marginTop: 20,
    elevation: 2,

  },
  btntext: {
    color: '#fafafa',
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  already: {
    color: "#555",
    marginBottom: 10,
    fontSize: 13,
    fontWeight: 'bold',
  },
  signin: {
    fontSize: 13,
    fontWeight: 'bold',
    color: '#01714b'
  },
  error: {
    fontSize: 11,
    color: 'red',
    alignSelf: 'flex-start',
    marginBottom: 8,
    marginLeft: 8,
  }
});
