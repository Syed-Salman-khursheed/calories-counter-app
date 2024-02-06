import AsyncStorage from '@react-native-async-storage/async-storage';
import React, {createContext, useEffect, useRef, useState} from 'react';

//Firebase
import auth from '@react-native-firebase/auth';
import storage from '@react-native-firebase/storage';
import firestore from '@react-native-firebase/firestore';
import {GeneralUtil} from '../Util';

const AuthContextProvider = ({children}) => {
  const [user, setUser] = useState(false);
  const [loading, setLoading] = useState({});
  const [userInfo, setUserInfo] = useState({});
  const [calories, setCalories] = useState([]);
  const [feedbacks, setFeedbacks] = useState([]);

  useEffect(() => {
    userRegistry();
  }, []);

  const userRegistry = async () => {
    const getUser = await AsyncStorage.getItem('User');

    if (getUser !== null) {
      const verification = auth().currentUser.emailVerified;
      console.log('Working from Registry');

      setUser(JSON.parse(getUser));
      AsyncStorage.setItem('EmailVerified', JSON.stringify(verification));
    }
  };

  const onSignIn = async (email, password, navigation) => {
    console.log(email,password)
    await auth()
      .signInWithEmailAndPassword(email, password)
      .then(Token => {
        const User = Token.user;
        const verification = User.emailVerified;
        AsyncStorage.setItem('EmailVerified', JSON.stringify(verification));

        firestore()
          .collection('Users')
          .doc(User.uid)
          .get()
          .then(async doc => {
            const UserData = doc.data();
            setUser(UserData);
            await AsyncStorage.setItem('User', JSON.stringify(UserData));
            getUserInfo(navigation)
            alert('Successfully Login !')
            // console.log('UserData => ', UserData);
            // navigation.navigate('Registration');
          });
      })
      .catch(error => {
        console.log(error.code,'logiiiin--->>')
        error.code === 'auth/invalid-login'
          ? alert('Your credentials are invalid!')
          : error.code === 'auth/wrong-password'
          ? alert('Your credentials are invalid!')
          : error.code === 'auth/network-request-failed'
          ? alert('Please check Your Internet Connection!')
          : console.error(error);
      });
  };

  const onSignUp = async (obj, navigation) => {
    const {
      email,
      password,
      firstName,
      lastName,
    } = obj;
console.log( email,
  password,
  firstName,
  lastName)
    await auth()
      .createUserWithEmailAndPassword(email, password)
      .then(Token => {
        const User = Token.user;

        // const verification = User.emailVerified;
        // AsyncStorage.setItem('EmailVerified', JSON.stringify(verification));

        // User.updateProfile({
        //   displayName: `${firstName} ${lastName}`,
        // });

        const data = {
          email: User.email,
          firstname: firstName,
          lastname: lastName,
          uid: User.uid,
          createdAt: GeneralUtil.datetimeFormatter(new Date()),
         
        };

        firestore()
          .collection('Users')
          .doc(User.uid)
          .set(data)
          .then(() => {
            alert('Successfully Sign Up !');
            // AsyncStorage.setItem('User', JSON.stringify(data));
            // setUser(data);
            
            navigation.navigate('Login');
          });
      })
      .catch(error => {
        error.code === 'auth/email-already-in-use'
          ? alert('That email address is already in use!')
          : error.code === 'auth/invalid-email'
          ? alert('That email address is invalid!')
          : console.error(error);
      });
  };

  const adduserInfo = async (obj, calories, navigation) => {
    console.log(obj, calories);
    const {
      activity,
      gender,
      goal,
      age,
      currentWeight,
      kgToGainLose,
      height,
      months,
    } = obj;
  
    try {
      const Token = await auth(); // Assuming auth() returns a promise
      const User = Token._user;
      console.log(User, 'userrrr--->>>');
  
      const data = {
        activity,
        gender,
        goal,
        age,
        currentWeight,
        kgToGainLose,
        height,
        months,
        perDayCalories: calories,
        uid: User?.uid,
        createdAt: GeneralUtil.datetimeFormatter(new Date()),
      };
  
      await firestore()
        .collection('UserInfo')
        .doc(User?.uid)
        .set(data);
        setUserInfo(data)
        await AsyncStorage.setItem('userInfo', JSON.stringify(data));
      navigation.navigate('Home');
    } catch (error) {
      alert(error);
    }
  };

  const getUserInfo = async (navigation) => {
    try{
      const Token = await auth(); // Assuming auth() returns a promise
      const User = Token._user;

      firestore()
        .collection('UserInfo')
        .doc(User.uid)
        .get()
        .then(async doc => {
          const userInfo =await doc.data();
          if(userInfo){
            console.log('get user ifo function --->>>>',userInfo)
            setUserInfo(userInfo)

            await AsyncStorage.setItem('userInfo', JSON.stringify(userInfo));

            navigation.navigate('Home');
          }
         else{
          navigation.navigate('Registration');
         }
        });

  }
  catch(error) {
      console.log(error,'error find user---->>>>')
      navigation.navigate('Registration');
    };

  }

  const addCalories = async (obj) => {
    const {
      itemName, calorie, image,weight
    } = obj;
  
    try {
      const Token = await auth(); // Assuming auth() returns a promise
      const User = Token._user;
  
      const data = {
        name:itemName, 
        calories:calorie,
         image,
         qty:weight,
        uid: User?.uid,
        createdAt: GeneralUtil.datetimeFormatter(new Date()),
      };
  
      await firestore()
        .collection('UserCalories')
        .add(data)
        .then(() => {
          alert('SuccessFully Added in Diet History');
         
        });
  
    } catch (error) {
      alert(error);
    }
  };

  const getCalories = async () => {
    console.log('get calories chala===>>>>>>>>>>>>>>>>>>>>>>')
    setLoading(true)
 try{
  const Token = await auth(); 
  const User = await Token._user;
  console.log(User.id,user?.uid)
  if (User?.uid) {
    firestore()
      .collection('UserCalories')
      .where('uid', '==', User?.uid || user?.uid)
      .orderBy('createdAt', 'desc')
      .onSnapshot(docSnap => {
        if (docSnap && !docSnap?.empty) {
          const dataSet = docSnap.docs.map(data => {
            const caloriesData = data.data();
            return {
              calories_id: data.id,
              calories: caloriesData.calories,
              image: caloriesData.image,
              name: caloriesData.name,
              qty: caloriesData.qty,
              createdAt:caloriesData?.createdAt
            };
          });
          console.log('calories----->>>',dataSet)
          setLoading(false)
          setCalories(dataSet);
        }})
  }
 } catch (error) {
  setLoading(false)

  alert(error);
}
  };

  const getTodayCalories = async () => {
  
    try {
      const Token = await auth(); // Assuming auth() returns a promise
      const User = Token._user;
  
      const currentDate = new Date();
  
      firestore()
        .collection('UserCalories')
        .where('uid', '==', User?.uid)
        // .where('createdAt', '<=', GeneralUtil.datetimeFormatter(new Date()))
        // .where('createdAt', '<=', GeneralUtil.datetimeFormatter(new Date() - 1))
        // .where('createdAt', '<', new Date(currentDate.getFullYear(), currentDate.getMonth(), currentDate.getDate() + 1))
        .onSnapshot(docSnap => {
          if (docSnap && !docSnap?.empty) {
            const dataSet = docSnap.docs.map(data => {
              const caloriesData = data.data();
              return {
                calories_id: data.id,
                calories: caloriesData.calories,
                image: caloriesData.image,
                name: caloriesData.name,
                qty: caloriesData.qty,
                createdAt:caloriesData?.createdAt
              };
            });
            console.log('today calories----------------------------------------------------------------->>>',dataSet)
            // setCalories(dataSet);
          }})
    } catch (error) {
      console.log('today calorie-------------------------------------------------------------------->>>>',error)
      alert(error);
    }
  };

  const addFeedback = async (message) => {
    try {
      const getUser = await AsyncStorage.getItem('User');
      const user=JSON.parse(getUser)
    
      const data = {
        firstname:user?.firstname, 
        lastname:user?.lastname, 
        email:user?.email, 
        message:message,
        uid: user?.uid,
        createdAt: GeneralUtil.datetimeFormatter(new Date()),
      };
  
      await firestore()
        .collection('UserFeedback')
        .add(data)
        .then(() => {
          alert('Feedback is Submitted');
         
        });
  
    } catch (error) {
      alert(error);
    }
  };
  
  const getFeedback = async () => {
 try{
  const Token = await auth(); 
  const user = Token._user;
  if (user.uid) {
    firestore()
      .collection('UserFeedback')
      .orderBy('createdAt', 'desc')
      .onSnapshot(docSnap => {
        if (docSnap && !docSnap?.empty) {
          const dataSet = docSnap.docs.map(data => {
            const UserFeedback = data.data();
            return {
              feedback_id: data.id,
              message: UserFeedback?.message,
              firstname: UserFeedback?.firstname,
              lastname: UserFeedback?.lastname,
              createdAt:UserFeedback?.createdAt
            };
          });
          setFeedbacks(dataSet);
        }})
  }
 } catch (error) {
  alert(error);
}
  };

  const recoverPass = email => {
    auth()
      .sendPasswordResetEmail(email)
      .then(() => {
        alert('Password Link send to your Email!');
      });
  };

  const onSignOut = async navigation => {
    console.log('logout---->>>>')
    setCalories([])
    AsyncStorage.removeItem('User');
    AsyncStorage.clear();
    auth()
      .signOut()
      .then(() => {
        navigation.navigate('Login');
        alert('Logout');
      });
  };

  return (
    <AuthContext.Provider
      value={{
        loading,
        user,
        userInfo,
        calories,
        feedbacks
      }}>
      <AuthAction.Provider value={{onSignOut, onSignUp, onSignIn,adduserInfo,addCalories,getCalories,addFeedback,getFeedback,getTodayCalories,getUserInfo}}>
        {children}
      </AuthAction.Provider>
    </AuthContext.Provider>
  );
};

export default AuthContextProvider;

// Auth Context
export const AuthContext = createContext();
export const AuthAction = createContext();