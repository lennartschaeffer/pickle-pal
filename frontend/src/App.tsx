import { MantineProvider } from "@mantine/core";
import Layout from "./components/Layout";
import Router from "./Router";
import "@mantine/core/styles.css";

function App() {
  return (
    <MantineProvider defaultColorScheme="dark">
      <Layout>
        <Router />
      </Layout>
    </MantineProvider>
  );
}

export default App;
