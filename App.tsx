/**
 * @format
 */

import {NavigationContainer} from '@react-navigation/native';
import 'react-native-gesture-handler';
import Rounting from './src/Config/Rounting';
import ContextProvider from './src/Context/ContextProvider';

function App() {
  return (
      <ContextProvider>

    <NavigationContainer>
      <Rounting />
    </NavigationContainer>
      </ContextProvider>
  );
}

export default App;
